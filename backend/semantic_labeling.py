"""Align Grounding DINO token evidence to an explicitly declared vocabulary.

This module never reads images, annotations, detector boxes, or model weights.
It uses each candidate's own exact prompt occurrence, not decoded fragments or
substring matching. Defaults are predeclared for the teacher-label repair:
mean sigmoid probability over lexical tokens, score >= .20, and a >= .025
margin between distinct candidate identities. These are not calibrated class
probabilities or thresholds selected against an evaluation set.
"""
from __future__ import annotations

import hashlib
import math
import unicodedata

import numpy as np


DEFAULT_MIN_SCORE = .20
DEFAULT_MIN_MARGIN = .025
SCORE_DEFINITION = "arithmetic_mean_sigmoid_over_candidate_lexical_tokens"


def _array(value):
    """Accept NumPy/list or detached framework tensors without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _vector(value, name):
    result = _array(value)
    if result.ndim == 2 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 1:
        raise ValueError(f"{name} must describe exactly one token sequence")
    return result


def _lexical(character):
    # Combining marks belong to the lexical character they modify. Delimiter
    # dots, hyphens, whitespace, and standalone punctuation carry no class score.
    return unicodedata.category(character)[0] in {"L", "N", "M"}


def _candidate_spans(prompt, candidate_labels):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if isinstance(candidate_labels, str):
        raise ValueError("candidate_labels must be a sequence, not a prompt string")
    labels = list(candidate_labels)
    if not labels or any(not isinstance(label, str) for label in labels):
        raise ValueError("candidate_labels must contain nonempty strings")
    labels = [label.strip(". \t\r\n") for label in labels]
    if any(not label or not any(_lexical(char) for char in label) for label in labels):
        raise ValueError("each candidate needs a nonempty lexical span")
    spans, cursor = [], 0
    while cursor < len(prompt) and prompt[cursor].isspace():
        cursor += 1
    for index, label in enumerate(labels):
        end = cursor + len(label)
        # Case changes are accepted only when they preserve the exact character
        # span length. Unicode normalization never changes prompt offsets.
        if prompt[cursor:end].casefold() != label.casefold():
            raise ValueError(f"candidate {index} does not match its declared prompt position")
        spans.append((cursor, end))
        cursor = end
        while cursor < len(prompt) and prompt[cursor].isspace():
            cursor += 1
        if index + 1 < len(labels) and (cursor >= len(prompt) or prompt[cursor] != "."):
            raise ValueError("candidate phrases must be separated by periods in declaration order")
        # A trailing period is optional; repeated period/space separators are
        # unambiguous because the preceding candidate matched in its entirety.
        while cursor < len(prompt) and (prompt[cursor] == "." or prompt[cursor].isspace()):
            cursor += 1
    if cursor != len(prompt):
        raise ValueError("prompt contains text outside the declared candidate phrases")
    return labels, spans


def _boolean_mask(value, length, name, default):
    if value is None:
        return np.full(length, default, dtype=bool)
    value = _vector(value, name)
    if value.shape != (length,) or value.dtype.kind not in "buif" or not np.isfinite(value).all() or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must contain {length} boolean values")
    return value.astype(bool)


def align_query_labels(token_logits, prompt, candidate_labels, offset_mapping,
                       attention_mask=None, special_tokens_mask=None, *,
                       min_score=DEFAULT_MIN_SCORE, min_margin=DEFAULT_MIN_MARGIN):
    """Return query-to-candidate scores and an unambiguous declared label or None.

    token_logits: [queries, model_text_width]. offset_mapping: [input_tokens,2],
    with Python character offsets into the exact prompt. Model output width may
    exceed input token count (e.g. DINO's padded width 256); extra columns are
    ignored. Attention=0, special tokens, and punctuation-only tokens are ignored.
    Every lexical character of each candidate must have usable token coverage;
    a truncated/empty candidate is an error, never a confident partial label.

    Score is the arithmetic mean of sigmoid(logit) at that candidate's own
    lexical token positions. A token with punctuation and letters is included;
    a punctuation-only token is excluded. No longest-substring heuristic, word
    selection by query score, or synonym inference is performed.

    scores[Q,C] retains each declared occurrence. Repeated identical labels
    (case/whitespace normalized only for identity) are one identity for ranking:
    use their maximum occurrence score and return the first declaration index.
    occurrence_indices identifies which repeated occurrence supplied that score.
    Different labels, including 'coffee'/'coffee mug', compete normally.

    candidate_indices=-1 and query_labels=None indicate rejection. ambiguous
    means a high-enough score without a clear distinct-identity margin; low-score
    rejection is separate. A single identity's runner-up score is defined as 0.
    NaNs are invalid, including padding. +/-infinity logits are valid saturated
    probabilities (DINO commonly masks padding with -infinity).
    """
    for name, value in (("min_score", min_score), ("min_margin", min_margin)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0,1]")
    labels, spans = _candidate_spans(prompt, candidate_labels)
    logits = _array(token_logits)
    if logits.ndim != 2 or logits.dtype.kind not in "buif" or np.isnan(logits).any():
        raise ValueError("token_logits must have numeric shape [queries,text_width] without NaNs")
    offsets = _array(offset_mapping)
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.dtype.kind not in "iu" or not len(offsets):
        raise ValueError("offset_mapping must have nonempty integer shape [input_tokens,2]")
    if len(offsets) > logits.shape[1]:
        raise ValueError("model text logits are shorter than the verified token input")
    if (offsets < 0).any() or (offsets[:, 1] < offsets[:, 0]).any() or (offsets[:, 1] > len(prompt)).any():
        raise ValueError("offset_mapping has invalid prompt character bounds")
    attention = _boolean_mask(attention_mask, len(offsets), "attention_mask", True)
    special = _boolean_mask(special_tokens_mask, len(offsets), "special_tokens_mask", False)
    lexical_positions = {index for index, char in enumerate(prompt) if _lexical(char)}
    token_positions = [{position for position in range(int(start), int(end)) if position in lexical_positions}
                       for start, end in offsets]
    token_indices = []
    for candidate, (start, end) in enumerate(spans):
        needed = {position for position in range(start, end) if position in lexical_positions}
        selected, covered = [], set()
        for token, positions in enumerate(token_positions):
            if not attention[token] or special[token] or not positions.intersection(needed):
                continue
            if not positions.issubset(needed):
                raise ValueError("a token crosses a candidate phrase boundary; alignment is unsafe")
            selected.append(token)
            covered.update(positions)
        if not selected or covered != needed:
            raise ValueError(f"candidate {candidate} has an empty or truncated usable token span")
        token_indices.append(selected)
    values = logits.astype(np.float64, copy=False)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    probabilities[~positive] = exponential / (1 + exponential)
    scores = np.stack([probabilities[:, indices].mean(axis=1) for indices in token_indices], axis=1)
    identities, occurrence_groups = [], []
    for index, label in enumerate(labels):
        identity = " ".join(label.casefold().split())
        if identity not in identities:
            identities.append(identity)
            occurrence_groups.append([])
        occurrence_groups[identities.index(identity)].append(index)
    query_count = len(logits)
    indices = np.full(query_count, -1, dtype=np.int64)
    occurrence_indices = np.full(query_count, -1, dtype=np.int64)
    best_scores, second_scores = np.zeros(query_count), np.zeros(query_count)
    margins = np.zeros(query_count)
    ambiguous = np.zeros(query_count, dtype=bool)
    query_labels, reasons = [None] * query_count, ["low_score"] * query_count
    group_scores = np.stack([scores[:, group].max(axis=1) for group in occurrence_groups], axis=1)
    for query in range(query_count):
        order = np.argsort(-group_scores[query], kind="stable")
        winner = int(order[0])
        best_scores[query] = group_scores[query, winner]
        second_scores[query] = group_scores[query, order[1]] if len(order) > 1 else 0.
        margins[query] = best_scores[query] - second_scores[query]
        if best_scores[query] < min_score:
            continue
        if margins[query] < min_margin or margins[query] == 0:
            ambiguous[query] = True
            reasons[query] = "ambiguous_margin"
            continue
        group = occurrence_groups[winner]
        canonical = group[0]
        occurrence = group[int(np.argmax(scores[query, group]))]
        indices[query] = canonical
        occurrence_indices[query] = occurrence
        query_labels[query] = labels[canonical]
        reasons[query] = "accepted"
    return {"scores": scores, "candidate_indices": indices, "occurrence_indices": occurrence_indices,
            "query_labels": query_labels, "ambiguous": ambiguous, "reasons": reasons,
            "best_scores": best_scores, "second_scores": second_scores, "margins": margins,
            "candidate_labels": labels, "candidate_spans": spans, "token_indices": token_indices,
            "duplicate_identity_groups": [group for group in occurrence_groups if len(group) > 1],
            "score_definition": SCORE_DEFINITION, "min_score": float(min_score), "min_margin": float(min_margin),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}


def align_grounding_dino_queries(processor, input_ids, token_logits, prompt, candidate_labels,
                                 attention_mask=None, *, min_score=DEFAULT_MIN_SCORE,
                                 min_margin=DEFAULT_MIN_MARGIN):
    """Obtain trustworthy offsets from the existing processor's exact tokenizer.

    Re-tokenize the original prompt without truncation, then require exact token
    ID equality with the model's input. Left/right padding is supported only when
    identified by attention_mask or the tokenizer's known pad_token_id. All
    padding positions are checked against that ID. Truncation, different special
    tokens, a different tokenizer/prompt, or offset support missing means failure.
    No tokenizer/model is downloaded or instantiated by this helper.

    Accepts input_ids [L] or [1,L], logits [Q,T] or [1,Q,T]; multi-image batches
    require a separate call for each image. Return fields include the verified
    actual-sequence offsets/masks, input ID digest, and tokenization_verified=True.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor must expose its existing tokenizer")
    if getattr(tokenizer, "is_fast", True) is False:
        raise ValueError("exact offset mapping requires the processor's fast tokenizer")
    ids = _vector(input_ids, "input_ids")
    if not len(ids) or ids.dtype.kind not in "iu" or (ids < 0).any():
        raise ValueError("input_ids must contain nonnegative integer token IDs")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if attention_mask is None:
        attention = ids != pad_id if pad_id is not None else np.ones(len(ids), dtype=bool)
    else:
        attention = _boolean_mask(attention_mask, len(ids), "attention_mask", True)
    active = np.flatnonzero(attention)
    if not len(active) or not np.array_equal(active, np.arange(active[0], active[-1] + 1)):
        raise ValueError("input attention must identify one contiguous, nonempty token sequence")
    if (~attention).any() and (pad_id is None or not np.all(ids[~attention] == pad_id)):
        raise ValueError("masked input IDs do not match the tokenizer's padding token")
    try:
        encoded = tokenizer(prompt, add_special_tokens=True, padding=False, truncation=False,
                            return_attention_mask=True, return_special_tokens_mask=True, return_offsets_mapping=True)
    except (TypeError, NotImplementedError, ValueError) as exc:
        raise ValueError("processor tokenizer could not provide exact prompt offsets") from exc
    if any(key not in encoded for key in ("input_ids", "offset_mapping", "attention_mask", "special_tokens_mask")):
        raise ValueError("tokenizer did not return input IDs, offsets, attention, and special-token masks")
    expected_ids = _vector(encoded["input_ids"], "retokenized input_ids")
    if expected_ids.dtype.kind not in "iu" or not np.array_equal(ids[active], expected_ids):
        raise ValueError("tokenization differs from model input IDs; refuse guessed or truncated alignment")
    expected_attention = _boolean_mask(encoded["attention_mask"], len(expected_ids), "retokenized attention_mask", True)
    if not expected_attention.all():
        raise ValueError("untruncated, unpadded retokenization unexpectedly contains masked tokens")
    offsets = _array(encoded["offset_mapping"])
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if offsets.shape != (len(expected_ids), 2) or offsets.dtype.kind not in "iu":
        raise ValueError("retokenized offsets do not match token IDs")
    special = _boolean_mask(encoded["special_tokens_mask"], len(expected_ids), "retokenized special_tokens_mask", False)
    actual_offsets = np.zeros((len(ids), 2), dtype=np.int64)
    actual_offsets[active] = offsets
    actual_special = np.ones(len(ids), dtype=bool)
    actual_special[active] = special
    logits = _array(token_logits)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    result = align_query_labels(logits, prompt, candidate_labels, actual_offsets, attention, actual_special,
                                min_score=min_score, min_margin=min_margin)
    result.update(tokenization_verified=True, input_token_count=len(ids), unpadded_token_count=len(active),
                  input_ids_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest(),
                  offset_mapping=actual_offsets, attention_mask=attention, special_tokens_mask=actual_special,
                  tokenizer_class=type(tokenizer).__name__)
    return result
