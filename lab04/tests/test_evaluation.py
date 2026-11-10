import unittest

from lab04.evaluation import score_ratings


class EvaluationTests(unittest.TestCase):
    def test_invalid_is_wrong_for_accuracy_and_f1_but_excluded_from_mae(self):
        result = score_ratings([1, 2, 3, 4, 5], [1, 3, None, 4, 1])
        self.assertEqual(result["accuracy"], 2 / 5)
        self.assertEqual(result["mae_stars"], 5 / 4)
        self.assertEqual(result["invalid"], 1)
        self.assertAlmostEqual(result["f1_macro"], (2 / 3 + 1) / 5)
        self.assertEqual(result["confusion_matrix"]["3"]["invalid"], 1)

    def test_constant_prediction_does_not_receive_high_macro_f1(self):
        result = score_ratings([1, 2, 3, 4, 5], [4] * 5)
        self.assertEqual(result["accuracy"], 0.2)
        self.assertAlmostEqual(result["f1_macro"], (1 / 3) / 5)

    def test_no_valid_answers(self):
        result = score_ratings([1, 5], [None, None])
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["f1_macro"], 0)
        self.assertIsNone(result["mae_stars"])


if __name__ == "__main__":
    unittest.main()
