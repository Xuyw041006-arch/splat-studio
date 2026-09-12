"""Canonical detector aliases stay bound to verified token spans, entirely on CPU."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import semantics
from tests.test_semantic_labeling import TinyTokenizer, tokenized


CANONICAL = ['coffee', 'coffee mug', 'tea in a glass']
QUERIES = ['coffee', 'coffee mug', 'glass of tea']
ALIASES = {'tea in a glass': 'glass of tea'}


class CpuBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        assert str(device) == 'cpu'
        return self


def detector_fixture(queries=QUERIES):
    """Four boxes: distinct coffee, mug, tea, and an ambiguous coffee/tea box."""
    prompt = '. '.join(queries) + '.'
    encoded = tokenized(prompt)
    count = len(encoded['input_ids'])
    probabilities = np.full((4, count), .01, dtype=np.float64)
    # Query spans, independent of decoded detector fragments.
    ranges = []
    offset = 0
    for query in queries:
        start = prompt.index(query, offset)
        end = start + len(query)
        ranges.append([i for i, (a, b) in enumerate(encoded['offset_mapping'])
                       if a < b and a >= start and b <= end and prompt[a:b].isalnum()])
        offset = end
    for index in range(3):
        probabilities[index, ranges[index]] = .8 - index * .05
    probabilities[3, ranges[0]] = .35
    probabilities[3, ranges[2]] = .34
    logits = torch.tensor(np.log(probabilities / (1 - probabilities)), dtype=torch.float32)[None]
    batch = CpuBatch(input_ids=torch.tensor([encoded['input_ids']]),
                     attention_mask=torch.tensor([encoded['attention_mask']]))
    out = SimpleNamespace(logits=logits)
    detected = {'boxes': torch.tensor([[0., 0., 8., 8.]] * 4),
                'scores': logits[0].sigmoid().amax(-1),
                # No output identity may be guessed from these fragments.
                'text_labels': ['mug', 'coffee', 'glass', 'tea coffee']}
    return SimpleNamespace(tokenizer=TinyTokenizer()), batch, out, detected, prompt


def test_alias_only_replaces_requested_query_and_preserves_canonical_order():
    labels = list(CANONICAL)
    canonical, queries = semantics._detector_query_vocabulary(labels, ALIASES)
    assert canonical == CANONICAL
    assert queries == QUERIES
    assert labels == CANONICAL
    assert ALIASES == {'tea in a glass': 'glass of tea'}


@pytest.mark.parametrize('aliases', [None, {}])
def test_no_alias_keeps_existing_query_vocabulary(aliases):
    canonical, queries = semantics._detector_query_vocabulary(CANONICAL, aliases)
    assert canonical == CANONICAL
    assert queries == CANONICAL


@pytest.mark.parametrize('aliases', [
    ['glass of tea'], 'glass of tea', 1,
    {'glass': 'glass of tea'},
    {'Tea in a glass': 'glass of tea'},
    {'tea in a glass': ''},
    {'tea in a glass': '   '},
    {'tea in a glass': None},
    {'tea in a glass': 3},
    {'tea in a glass': 'coffee'},
    {'coffee': 'mug', 'coffee mug': 'mug'},
    {'coffee': 'MUG', 'coffee mug': 'mug'},
])
def test_invalid_unknown_empty_or_ambiguous_aliases_are_rejected(aliases):
    with pytest.raises((ValueError, TypeError)):
        semantics._detector_query_vocabulary(CANONICAL, aliases)


def test_verified_query_positions_return_canonical_tea_and_keep_similar_classes_separate():
    processor, batch, out, detected, prompt = detector_fixture()
    assigned, metadata, diagnostics = semantics._aligned_detector_labels(
        processor, batch, out, detected, prompt, QUERIES, .3, canonical_labels=CANONICAL)
    assert assigned == ['coffee', 'coffee mug', 'tea in a glass', None]
    assert [row['detector_query_index'] for row in metadata] == [0, 1, 2, 3]
    assert metadata[2]['raw_detector_label'] == 'glass'
    assert metadata[2]['detector_query_label'] == 'glass of tea'
    assert metadata[2]['canonical_label'] == 'tea in a glass'
    assert metadata[2]['detector_query_alias_applied'] is True
    assert metadata[0]['detector_query_alias_applied'] is False
    assert metadata[1]['canonical_label'] == 'coffee mug'
    assert diagnostics['accepted_labels'] == 3
    assert diagnostics['rejected_ambiguous'] == 1
    assert diagnostics['missing_candidate_labels'] == []


def test_aliases_do_not_bypass_actual_input_token_verification():
    processor, batch, out, detected, prompt = detector_fixture()
    batch.input_ids[0, 1] += 1
    with pytest.raises(ValueError, match='differs'):
        semantics._aligned_detector_labels(processor, batch, out, detected,
            prompt, QUERIES, .3, canonical_labels=CANONICAL)


def test_aliases_do_not_bypass_declared_query_span_validation():
    processor, batch, out, detected, prompt = detector_fixture()
    # Same model input, but declaring the longer canonical label as the query
    # would associate evidence with the wrong prompt span.
    with pytest.raises(ValueError, match='declared prompt position'):
        semantics._aligned_detector_labels(processor, batch, out, detected,
            prompt, CANONICAL, .3, canonical_labels=CANONICAL)


def test_aliases_do_not_bypass_postprocessor_box_query_order_verification():
    processor, batch, out, detected, prompt = detector_fixture()
    detected['scores'] = detected['scores'].flip(0)
    with pytest.raises(ValueError, match='query order'):
        semantics._aligned_detector_labels(processor, batch, out, detected,
            prompt, QUERIES, .3, canonical_labels=CANONICAL)


def test_aligned_helper_without_canonical_argument_keeps_old_behavior():
    processor, batch, out, detected, prompt = detector_fixture(CANONICAL)
    assigned, metadata, diagnostics = semantics._aligned_detector_labels(
        processor, batch, out, detected, prompt, CANONICAL, .3)
    assert assigned == [*CANONICAL, None]
    assert metadata[2]['raw_detector_label'] == 'glass'
    assert metadata[2]['label_alignment_source'] == 'declared_query_token_mean_v1'
    assert diagnostics['accepted_labels'] == 3


def test_local_inventory_refuses_aliases_without_strict_alignment_before_model_load(monkeypatch):
    def forbid_load(*args, **kwargs):
        pytest.fail('Invalid alias mode must be rejected before any model loading or download')
    monkeypatch.setattr(semantics, '_load_local_models', forbid_load)
    with pytest.raises(ValueError, match='strict_query_labels'):
        semantics._local_inventory([], {'device': 'cpu', 'candidate_labels': CANONICAL,
            'detector_query_aliases': ALIASES, 'strict_query_labels': False})


def test_local_inventory_exports_canonical_mask_labels_and_alias_provenance(monkeypatch, tmp_path):
    processor_info, batch, out, detected, expected_prompt = detector_fixture()

    class Processor:
        tokenizer = processor_info.tokenizer

        def __call__(self, *, images, text, return_tensors):
            assert text == expected_prompt
            assert return_tensors == 'pt'
            return batch

        def post_process_grounded_object_detection(self, output, input_ids, *,
                                                  threshold, text_threshold, target_sizes):
            assert output is out
            assert threshold == .3
            return [detected]

    class SamProcessor:
        def __call__(self, *, images, input_boxes, return_tensors):
            assert len(input_boxes[0]) == 3  # Ambiguous fourth detection is dropped.
            return {'original_sizes': torch.tensor([[16, 16]]),
                    'reshaped_input_sizes': torch.tensor([[16, 16]])}

        def post_process_masks(self, pred_masks, original_sizes, reshaped_input_sizes):
            return [torch.ones((3, 1, 16, 16), dtype=torch.bool)]

    def load_models(model_id, sam_id, device, local_only, cache_dir):
        assert device == 'cpu'
        assert local_only is True
        return Processor(), lambda **kwargs: out, SamProcessor(), lambda **kwargs: SimpleNamespace(
            pred_masks=torch.ones((1, 3, 1, 16, 16)))

    monkeypatch.setattr(semantics, '_load_local_models', load_models)
    image_path = tmp_path / 'train.jpg'
    Image.new('RGB', (16, 16), color=(40, 80, 120)).save(image_path)
    result = semantics._local_inventory([image_path], {'device': 'cpu',
        'candidate_labels': CANONICAL, 'detector_query_aliases': ALIASES,
        'strict_query_labels': True, 'allow_model_download': False})
    records = result['masks']
    assert {row['label'] for row in records} == set(CANONICAL)
    tea = next(row for row in records if row['label'] == 'tea in a glass')
    assert tea['detector_query_label'] == 'glass of tea'
    assert tea['canonical_label'] == 'tea in a glass'
    assert tea['detector_query_alias_applied'] is True
    assert tea['raw_detector_label'] == 'glass'
    assert tea['mask'].shape == (16, 16)
    assert tea['image_name'] == 'train.jpg'
    assert result['image_coverage'][0]['mask_count'] == 3
