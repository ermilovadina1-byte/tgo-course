"""Проверки отбора данных до подключения языковой модели."""

import unittest

from datasets import Dataset, DatasetDict

from lab04.data import balanced_indices, select_reviews, star_counts, summarize


class ReviewDataTests(unittest.TestCase):
    def test_balanced_selection_is_reproducible_without_replacement(self):
        labels = [0] * 10 + [1] * 20 + [2] * 30 + [3] * 40 + [4] * 50
        indices = balanced_indices(labels, 5, 42)
        self.assertEqual(indices, balanced_indices(labels, 5, 42))
        self.assertNotEqual(indices, balanced_indices(labels, 5, 43))
        self.assertEqual(len(set(indices)), 25)
        self.assertEqual(star_counts(labels[i] for i in indices), dict.fromkeys("12345", 5))

    def test_cannot_fill_missing_class_with_duplicates(self):
        with self.assertRaisesRegex(ValueError, "оценки 5"):
            balanced_indices([0, 1, 2, 3], 1, 42)

    def test_test_prefix_and_few_shot_are_independent_of_training_seed(self):
        def split(name, n):
            return Dataset.from_dict({"text": [f"{name} {i}" for i in range(n)],
                                      "label": [i % 5 for i in range(n)]})
        data = DatasetDict({"train": split("train", 100),
                            "validation": split("validation", 10),
                            "test": split("test", 1000)})
        a = select_reviews(data, seed=42, per_class=5)
        b = select_reviews(data, seed=43, fast=True, per_class=5)
        self.assertEqual(a.test_indices, list(range(1000)))
        self.assertEqual(b.test_indices, a.test_indices[:24])
        self.assertEqual(a.few_shot_indices, b.few_shot_indices)
        report = summarize(data, a, source_revision=None)
        self.assertEqual(report["exact_text_overlap"],
                         {"train_test": 0, "few_shot_test": 0, "train_few_shot": 0})
        self.assertIsNone(report["source_revision"])


if __name__ == "__main__":
    unittest.main()
