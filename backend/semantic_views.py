"""Training-view selection and byte-bound provenance for semantic teachers.

This module never discovers additional files or reads evaluation annotations.
The caller supplies the input split; explicit holdouts always win over an
allowlist. Basenames are accepted only when they identify one supplied file.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _catalog(image_paths):
    paths = sorted({str(Path(path).resolve()) for path in image_paths})
    names = {}
    for path in paths:
        names.setdefault(Path(path).name, []).append(path)
    return paths, names


def resolve_image_reference(reference, image_paths):
    """Resolve a reference inside a caller-owned image list, never by guesswork."""
    paths, names = _catalog(image_paths)
    value = str(reference)
    exact = str(Path(value).resolve())
    if exact in paths:
        return exact
    # An explicit foreign path cannot borrow an in-split file's basename.
    if Path(value).name != value:
        raise ValueError(f'Image path is outside the selected training inputs: {value}')
    matches = names.get(value, [])
    if len(matches) != 1:
        raise ValueError(f'Image name does not identify one training input: {value}')
    return matches[0]


def select_semantic_views(image_paths, config=None):
    """Return selected absolute paths and a deterministic split/coverage report.

    ``all`` is the default. ``sampled`` evenly selects up to semantic_max_views
    from the sorted eligible inputs. Explicit benchmark callers may continue
    supplying an already fixed subset; this function never expands that subset.
    """
    config = config or {}
    supplied = list(image_paths)
    paths, _ = _catalog(supplied)
    legacy_limit = 'semantic_view_mode' not in config and config.get('max_images') is not None
    mode = config.get('semantic_view_mode', 'sampled' if legacy_limit else 'all')
    if mode not in {'all', 'sampled'}:
        raise ValueError('semantic_view_mode must be all or sampled')
    allowed = config.get('allowed_image_paths')
    held_out = config.get('held_out_image_paths', [])
    if (allowed is not None and not isinstance(allowed, (list, tuple, set))) or not isinstance(held_out, (list, tuple, set)):
        raise ValueError('Training allowlist and held-out images must be arrays')
    allowed_set = set(paths) if allowed is None else {resolve_image_reference(p, paths) for p in allowed}
    heldout_set = {resolve_image_reference(p, paths) for p in held_out}
    eligible = [p for p in paths if p in allowed_set and p not in heldout_set]
    selected = list(eligible)
    if mode == 'sampled':
        maximum = config.get('semantic_max_views', config.get('max_images', 24) if legacy_limit else 24)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 300:
            raise ValueError('semantic_max_views must be an integer in [1, 300]')
        if len(eligible) > maximum:
            if legacy_limit:
                eligible_set = set(eligible)
                selected = list(dict.fromkeys(str(Path(p).resolve()) for p in supplied
                    if str(Path(p).resolve()) in eligible_set))[:maximum]
            else:
                indices = np.linspace(0, len(eligible) - 1, maximum, dtype=int)
                selected = [eligible[i] for i in indices]
    selected_set = set(selected)
    excluded = [{'image_path': path, 'reason': 'held_out' if path in heldout_set else
                 'not_in_training_allowlist' if path not in allowed_set else 'sampled_out'}
                for path in paths if path not in selected_set]
    return selected, {
        'policy': mode, 'selection_method': 'legacy_prefix_limit' if legacy_limit else 'all' if mode == 'all' else 'evenly_spaced',
        'split_source': 'explicit_training_allowlist' if allowed is not None else 'caller_supplied_inputs',
        'input_count': len(supplied), 'unique_input_count': len(paths), 'eligible_count': len(eligible),
        'selected_count': len(selected), 'analyzed_count': 0, 'masked_count': 0,
        'selected_images': selected, 'excluded_images': excluded, 'images': [],
        'all_eligible_selected': len(selected) == len(eligible),
        'all_selected_analyzed': False,
        'evaluation_annotations_used': False,
    }


def validate_mask_sources(records, image_paths, require_hash=False):
    """Reject stale/wrong-file teachers; return copies with exact input paths.

    Legacy imported masks can lack hashes unless require_hash=True. Their lack
    of byte binding is explicit; a hash present on any record is always checked.
    """
    paths, _ = _catalog(image_paths)
    hashes, normalized = {}, []
    for original in records:
        record = dict(original)
        reference = record.get('image_path') or record.get('image_name') or record.get('frame_id')
        if not reference:
            raise ValueError('A semantic mask needs its source image reference')
        path = resolve_image_reference(reference, paths)
        for key in ('image_path', 'image_name', 'frame_id'):
            if record.get(key) is not None and resolve_image_reference(record[key], paths) != path:
                raise ValueError(f'Conflicting source image references on semantic mask: {key}')
        if path not in hashes:
            hashes[path] = file_sha256(path)
        expected = record.get('source_image_sha256')
        if expected is None and require_hash:
            raise ValueError(f'Mask source image SHA256 is missing: {Path(path).name}')
        if expected is not None and expected != hashes[path]:
            raise ValueError(f'Mask source image SHA256 mismatch: {Path(path).name}')
        if record.get('mask_sha256') is not None:
            if not record.get('mask_path') or file_sha256(record['mask_path']) != record['mask_sha256']:
                raise ValueError(f'Mask SHA256 mismatch: {Path(path).name}')
        record.update(image_path=path, image_name=Path(path).name,
                      source_image_binding='sha256_verified' if expected is not None else 'legacy_unbound')
        normalized.append(record)
    return normalized
