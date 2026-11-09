"""Проверки модели на маленьких примерах с ручным расчетом."""

import math
import unittest

from pt01.run import BigramModel


class BigramTests(unittest.TestCase):
    def setUp(self):
        self.model = BigramModel(alpha=1).fit(["кот", "спит", "кот", "ест"])

    def test_smoothed_probability_matches_hand_calculation(self):
        # Словарь: кот, спит, ест, <UNK>. Два продолжения после кот.
        self.assertEqual(self.model.pair_counts["кот", "спит"], 1)
        self.assertAlmostEqual(self.model.prob("кот", "спит"), 2 / 6)
        self.assertAlmostEqual(self.model.prob("кот", "кот"), 1 / 6)

    def test_rows_sum_to_one_including_unknown_and_terminal_context(self):
        for context in ("кот", "спит", "ест", "собака"):
            self.assertAlmostEqual(
                math.fsum(self.model.prob(context, word) for word in self.model.vocab),
                1.0,
            )
        # Единственное 'ест' стоит последним и не имеет наблюдаемых продолжений.
        self.assertEqual(self.model.context_counts["ест"], 0)

    def test_unknown_words_share_one_category_without_updating_counts(self):
        before = self.model.pair_counts.copy()
        self.assertEqual(self.model.prob("кот", "мышь"), self.model.prob("кот", "<UNK>"))
        self.assertEqual(self.model.prob("кот", "мышь"), self.model.prob("кот", "рыба"))
        self.model.perplexity(["кот", "мышь", "рыба"])
        self.assertEqual(before, self.model.pair_counts)
        self.assertNotIn("мышь", self.model.vocab)

    def test_logprob_and_perplexity_match_hand_calculation(self):
        # P(спит|кот)=1/3, P(кот|спит)=2/5, произведение=2/15.
        words = ["кот", "спит", "кот"]
        self.assertAlmostEqual(self.model.logprob(words), math.log(2 / 15))
        self.assertAlmostEqual(self.model.perplexity(words), math.sqrt(15 / 2))

    def test_refit_replaces_old_counts(self):
        self.model.fit(["она", "читает"])
        self.assertNotIn("кот", self.model.vocab)
        self.assertEqual(sum(self.model.pair_counts.values()), 1)

    def test_invalid_inputs(self):
        for alpha in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                BigramModel(alpha)
        with self.assertRaises(ValueError):
            BigramModel().prob("кот", "спит")
        with self.assertRaises(ValueError):
            self.model.perplexity(["кот"])


if __name__ == "__main__":
    unittest.main()
