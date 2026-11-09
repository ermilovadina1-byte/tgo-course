"""Ручные примеры BPE, частоты, границы слов и приоритет правил."""

import unittest
from collections import Counter

from pt01.run import BPETokenizer, merge_pair


class BPETests(unittest.TestCase):
    def test_lesson_example_with_weighted_counts(self):
        bpe = BPETokenizer().fit(Counter({'мама': 2, 'рама': 3, 'рамка': 1}), 5, 1)
        self.assertEqual(bpe.merges, [
            ('м', 'а'), ('ма', '</w>'), ('р', 'а'), ('ра', 'ма</w>'), ('ма', 'ма</w>')
        ])
        self.assertEqual(bpe.merge_frequencies, [7, 5, 4, 3, 2])
        self.assertEqual(bpe.encode_word('рамам'), ['ра', 'ма', 'м', '</w>'])
        self.assertEqual(bpe.encode_word('мама'), ['мама</w>'])

    def test_overlapping_pairs_merge_without_losing_letters(self):
        self.assertEqual(merge_pair(tuple('ааааа'), ('а', 'а')), ('аа', 'аа', 'а'))
        bpe = BPETokenizer().fit(Counter({'ааааа': 3}), 3, 1)
        for word in ('аа', 'аааа', 'ааааааа'):
            self.assertEqual(''.join(bpe.encode_word(word)), word + '</w>')

    def test_rank_takes_priority_over_leftmost_pair(self):
        bpe = BPETokenizer()
        bpe.ranks = {('б', 'в'): 0, ('а', 'б'): 1, ('а', 'бв'): 2}
        bpe.fitted = True
        self.assertEqual(bpe.encode_word('абв'), ['абв', '</w>'])

    def test_threshold_ties_and_refit(self):
        counts = Counter({'аб': 10, 'аж': 1})
        bpe = BPETokenizer().fit(counts, 1, 2)
        # Две пары с частотой 10: ('а', 'б') выбирается раньше ('б', '</w>').
        self.assertEqual(bpe.merges, [('а', 'б')])
        self.assertEqual(bpe.selected_types, 1)
        self.assertEqual(bpe.selected_occurrences, 10)
        self.assertIn('ж', bpe.symbols)
        self.assertEqual(bpe.encode_word('аж'), ['а', 'ж', '</w>'])
        reverse = BPETokenizer().fit(Counter(dict(reversed(list(counts.items())))), 1, 2)
        self.assertEqual(bpe.merges, reverse.merges)
        bpe.fit(Counter({'кот': 1}), 0, 1)
        self.assertEqual(bpe.merges, [])
        self.assertEqual(bpe.encode_word('кот'), ['к', 'о', 'т', '</w>'])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            BPETokenizer().encode_word('кот')
        for counts, merges, threshold in ((Counter(), 1, 1), (Counter({'кот': 1}), -1, 1),
                                          (Counter({'кот': 1}), 1, 0), (Counter({'кот': 1}), 1, 2)):
            with self.assertRaises(ValueError):
                BPETokenizer().fit(counts, merges, threshold)


if __name__ == '__main__':
    unittest.main()
