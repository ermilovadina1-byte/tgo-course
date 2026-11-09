"""Наши проверки подготовки данных. Не заменяют тесты преподавателя."""

import unittest

from pt01.run import split_words, tokenize_words


class PreparationTests(unittest.TestCase):
    def test_word_normalization(self):
        self.assertEqual(
            tokenize_words("Ёлка, КОТ! кто-то по-русски 123 English"),
            ["елка", "кот", "кто-то", "по-русски"],
        )

    def test_split_preserves_order_and_all_words(self):
        words = [str(i) for i in range(101)]
        train, val = split_words(words)
        self.assertEqual(len(train), 90)
        self.assertEqual(len(val), 11)
        self.assertEqual(train + val, words)
        self.assertEqual(train[-1], "89")
        self.assertEqual(val[0], "90")

    def test_too_short_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            split_words(["кот", "спит"])


if __name__ == "__main__":
    unittest.main()
