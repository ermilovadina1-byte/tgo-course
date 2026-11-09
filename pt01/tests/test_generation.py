"""Температура по ручному расчету и воспроизводимость генерации."""

import math
import random
import unittest

from pt01.run import BigramModel, temperature_distribution


class GenerationTests(unittest.TestCase):
    def test_temperature_matches_hand_calculation(self):
        for temperature, expected in ((0.5, [16 / 17, 1 / 17]),
                                      (1, [0.8, 0.2]),
                                      (2, [2 / 3, 1 / 3])):
            actual = temperature_distribution([0.8, 0.2], temperature)
            for got, want in zip(actual, expected):
                self.assertAlmostEqual(got, want)
            self.assertAlmostEqual(math.fsum(actual), 1)

    def test_zero_probability_and_small_temperature(self):
        self.assertEqual(temperature_distribution([0, 1], 0.5), [0, 1])
        self.assertEqual(temperature_distribution([0.8, 0.2], 1e-300), [1, 0])

    def test_invalid_inputs(self):
        model = BigramModel().fit(["кот", "спит"])
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                model.generate("кот", temperature=temperature)
        for n in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                model.generate("кот", n_words=n)
        with self.assertRaises(ValueError):
            model.generate("два слова")
        for probabilities in ([], [0, 0], [-1, 2], [float("nan")]):
            with self.assertRaises(ValueError):
                temperature_distribution(probabilities, 1)

    def test_reproducibility_length_and_no_training_mutation(self):
        model = BigramModel().fit(["кот", "спит", "кот", "ест"])
        before_counts = model.pair_counts.copy()
        before_rng = random.getstate()
        for temperature in (0.5, 1, 2):
            first = model.generate("кот", temperature=temperature, seed=42)
            second = model.generate("кот", temperature=temperature, seed=42)
            self.assertEqual(first, second)
            words = first.split()
            self.assertEqual(len(words), 30)
            self.assertEqual(words[0], "кот")
            self.assertTrue(set(words) <= model.vocab)
        self.assertEqual(model.pair_counts, before_counts)
        self.assertEqual(random.getstate(), before_rng)
        self.assertEqual(model.generate("кот", n_words=1), "кот")


if __name__ == "__main__":
    unittest.main()
