import re
from types import SimpleNamespace
import unittest

import numpy as np

from backend.semantic_labeling import align_grounding_dino_queries, align_query_labels


def logits(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.log(probabilities / (1 - probabilities))


class TinyTokenizer:
    """Transparent test tokenizer with real offsets; no model/image dependencies."""
    is_fast = True
    pad_token_id = 0

    def __call__(self, prompt, **kwargs):
        if kwargs.get("padding") is not False or kwargs.get("truncation") is not False:
            raise ValueError("tests require unpadded, untruncated tokenization")
        matches = list(re.finditer(r"[\w]+|[^\w\s]", prompt, re.UNICODE))
        return {"input_ids": [101] + list(range(1000, 1000 + len(matches))) + [102],
                "offset_mapping": [(0, 0)] + [match.span() for match in matches] + [(0, 0)],
                "attention_mask": [1] * (len(matches) + 2),
                "special_tokens_mask": [1] + [0] * len(matches) + [1]}


def tokenized(prompt):
    return TinyTokenizer()(prompt, padding=False, truncation=False)


class SemanticLabelingTests(unittest.TestCase):
    def test_candidate_positions_prevent_cross_phrase_or_longest_substring_assignment(self):
        prompt = "coffee. coffee mug."
        encoded = tokenized(prompt)
        # CLS, coffee, '.', coffee, mug, '.', SEP
        evidence = [[.99, .9, .99, .05, .05, .99, .99],
                    [.99, .05, .99, .8, .6, .99, .99]]
        result = align_query_labels(logits(evidence), prompt, ["coffee", "coffee mug"], encoded["offset_mapping"],
                                    special_tokens_mask=encoded["special_tokens_mask"])
        np.testing.assert_allclose(result["scores"], [[.9, .05], [.05, .7]])
        self.assertEqual(result["query_labels"], ["coffee", "coffee mug"])
        self.assertEqual(result["token_indices"], [[1], [3, 4]])
        self.assertEqual(result["candidate_spans"], [(0, 6), (8, 18)])

    def test_close_different_labels_return_none_and_low_score_is_separate(self):
        prompt = "cup. mug."
        result = align_query_labels(logits([[.99, .4, .99, .39, .99, .99],
                                            [.99, .1, .99, .01, .99, .99]]),
                                    prompt, ["cup", "mug"], tokenized(prompt)["offset_mapping"])
        self.assertEqual(result["query_labels"], [None, None])
        self.assertEqual(result["candidate_indices"].tolist(), [-1, -1])
        self.assertEqual(result["ambiguous"].tolist(), [True, False])
        self.assertEqual(result["reasons"], ["ambiguous_margin", "low_score"])

    def test_repeated_labels_are_one_identity_with_separate_occurrence_scores(self):
        prompt = "cup. cup. mug."
        encoded = tokenized(prompt)
        result = align_query_labels(logits([[.01, .2, .99, .8, .99, .1, .99, .01]]), prompt,
                                    ["cup", "cup", "mug"], encoded["offset_mapping"])
        np.testing.assert_allclose(result["scores"], [[.2, .8, .1]])
        self.assertEqual(result["query_labels"], ["cup"])
        self.assertEqual(result["candidate_indices"].tolist(), [0])
        self.assertEqual(result["occurrence_indices"].tolist(), [1])
        self.assertEqual(result["duplicate_identity_groups"], [[0, 1]])

    def test_hyphens_and_separator_punctuation_do_not_raise_phrase_score(self):
        prompt = "dall-e brand."
        encoded = tokenized(prompt)
        # CLS, dall, '-', e, brand, '.', SEP. Punctuation probabilities are high.
        result = align_query_labels(logits([[.99, .06, .99, .06, .06, .99, .99]]), prompt,
                                    ["dall-e brand"], encoded["offset_mapping"])
        np.testing.assert_allclose(result["scores"], [[.06]])
        self.assertEqual(result["token_indices"], [[1, 3, 4]])
        self.assertEqual(result["query_labels"], [None])

    def test_mean_uses_every_lexical_token_including_wordpiece(self):
        prompt = "stuffed bear."
        offsets = [(0, 0), (0, 5), (5, 7), (8, 12), (12, 13), (0, 0)]
        result = align_query_labels(logits([[.99, .9, .3, .3, .99, .99]]), prompt,
                                    ["stuffed bear"], offsets)
        np.testing.assert_allclose(result["scores"], [[.5]])
        self.assertEqual(result["token_indices"], [[1, 2, 3]])

    def test_truncated_or_cross_candidate_offsets_cannot_create_partial_label(self):
        prompt = "coffee mug."
        with self.assertRaisesRegex(ValueError, "truncated"):
            align_query_labels(logits([[.99, .99, .99]]), prompt, ["coffee mug"], [(0, 0), (0, 6), (0, 0)])
        with self.assertRaisesRegex(ValueError, "boundary"):
            align_query_labels(logits([[.99]]), "cup. mug.", ["cup", "mug"], [(0, 8)])
        with self.assertRaisesRegex(ValueError, "lexical span"):
            align_query_labels(logits([[.99]]), ".", ["."], [(0, 1)])

    def test_padding_and_infinite_masked_logits_are_ignored_but_nan_is_rejected(self):
        prompt = "cup."
        encoded = tokenized(prompt)
        evidence = np.array([[np.inf, 2., np.inf, np.inf, -np.inf, np.inf]])
        result = align_query_labels(evidence, prompt, ["cup"], encoded["offset_mapping"])
        self.assertAlmostEqual(result["best_scores"][0], 1 / (1 + np.exp(-2)))
        evidence[0, -1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaNs"):
            align_query_labels(evidence, prompt, ["cup"], encoded["offset_mapping"])
        with self.assertRaisesRegex(ValueError, "shorter"):
            align_query_labels(np.ones((2, 2)), prompt, ["cup"], encoded["offset_mapping"])

    def test_helper_verifies_ids_and_supports_left_and_right_padding(self):
        processor = SimpleNamespace(tokenizer=TinyTokenizer())
        prompt = "cup. mug."
        encoded = tokenized(prompt)
        original_ids = encoded["input_ids"]
        for before, after in ((0, 2), (2, 0)):
            ids = [0] * before + original_ids + [0] * after
            attention = [0] * before + [1] * len(original_ids) + [0] * after
            evidence = np.full((1, len(ids) + 4), -np.inf)
            evidence[0, before + 1] = logits(.8)
            evidence[0, before + 3] = logits(.1)
            result = align_grounding_dino_queries(processor, [ids], evidence[None], prompt, ["cup", "mug"],
                                                  attention_mask=[attention])
            self.assertEqual(result["query_labels"], ["cup"])
            self.assertTrue(result["tokenization_verified"])
            self.assertEqual(result["input_token_count"], len(ids))
            self.assertEqual(result["token_indices"], [[before + 1], [before + 3]])

    def test_helper_rejects_actual_input_mismatch_and_wrong_padding(self):
        processor = SimpleNamespace(tokenizer=TinyTokenizer())
        prompt = "cup."
        ids = tokenized(prompt)["input_ids"]
        wrong = list(ids)
        wrong[1] += 1
        for value in (wrong, ids[:-1]):
            with self.assertRaisesRegex(ValueError, "differs"):
                align_grounding_dino_queries(processor, value, np.zeros((1, 256)), prompt, ["cup"])
        with self.assertRaisesRegex(ValueError, "padding token"):
            align_grounding_dino_queries(processor, [77] + ids, np.zeros((1, 256)), prompt, ["cup"],
                                         attention_mask=[0] + [1] * len(ids))

    def test_prompt_order_empty_query_batch_and_unicode_offsets(self):
        prompt = "茶杯..  灯."
        encoded = tokenized(prompt)
        result = align_query_labels(np.empty((0, len(encoded["input_ids"]))), prompt,
                                    ["茶杯", "灯"], encoded["offset_mapping"])
        self.assertEqual(result["scores"].shape, (0, 2))
        self.assertEqual(result["query_labels"], [])
        with self.assertRaisesRegex(ValueError, "declared prompt position"):
            align_query_labels(np.zeros((1, 256)), prompt, ["灯", "茶杯"], encoded["offset_mapping"])


if __name__ == "__main__":
    unittest.main()
