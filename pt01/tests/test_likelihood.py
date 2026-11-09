"""Правдоподобие, сохранение слов при перемешивании и отсутствие дообучения."""

import math
import unittest
from collections import Counter

from pt01.run import BigramModel, compare_likelihood


class LikelihoodTests(unittest.TestCase):
    def test_fixed_fragment_and_hand_calculation(self):
        words = ["кот", "спит"] * 6
        model = BigramModel(alpha=1).fit(words)
        val = words + ["лишнее", "слово"]
        result = compare_likelihood(model, val)
        self.assertEqual(result["sentence"].split(), words)
        self.assertEqual(Counter(result["shuffled"].split()), Counter(words))
        self.assertEqual(result["predictions"], 11)
        # P(спит|кот)=7/9 шесть раз, P(кот|спит)=6/8 пять раз.
        expected = 6 * math.log(7 / 9) + 5 * math.log(6 / 8)
        self.assertAlmostEqual(result["logprob_sentence"], expected)
        self.assertEqual(result["unseen_pairs_in_sentence"], 0)
        self.assertEqual(result, compare_likelihood(model, val))
        self.assertEqual(val, words + ["лишнее", "слово"])

    def test_unseen_pairs_have_finite_scores_without_training(self):
        model = BigramModel().fit(["кот", "спит"])
        before = model.pair_counts.copy()
        result = compare_likelihood(model, ["кот", "рыба"] * 6)
        self.assertEqual(result["unseen_pairs_in_sentence"], 11)
        self.assertTrue(math.isfinite(result["logprob_sentence"]))
        self.assertTrue(math.isfinite(result["logprob_shuffled"]))
        self.assertEqual(model.pair_counts, before)
        self.assertNotIn("рыба", model.vocab)
        self.assertAlmostEqual(
            math.fsum(row["logprob"] for row in result["sentence_trace"]),
            result["logprob_sentence"],
        )

    def test_short_fragment_is_rejected(self):
        model = BigramModel().fit(["кот", "спит"])
        with self.assertRaises(ValueError):
            compare_likelihood(model, ["кот", "спит"])


if __name__ == "__main__":
    unittest.main()
