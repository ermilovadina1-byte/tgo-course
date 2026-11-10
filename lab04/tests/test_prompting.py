import unittest

from lab04.prompting import few_shot_messages, parse_rating, zero_shot_messages


class PromptingTests(unittest.TestCase):
    def test_only_task_and_review_are_sent(self):
        messages = zero_shot_messages("Цены конские.")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("Цены конские.", messages[1]["content"])

    def test_invalid_answers_are_not_silently_converted_to_ratings(self):
        for text in ("", "не знаю", "15", "0", "6", "4 или 5", "4/5", "3.5", "-1", "5 баллов из 10"):
            with self.subTest(text=text):
                self.assertIsNone(parse_rating(text))
        for text in ("5", " 5\n", "Оценка: 5", "5."):
            with self.subTest(text=text):
                self.assertEqual(parse_rating(text), 5)

    def test_few_shot_preserves_task_and_last_question(self):
        examples = [{"text": f"Учебный отзыв {i}", "label": i} for i in range(5)]
        zero = zero_shot_messages("Новый отзыв")
        few = few_shot_messages("Новый отзыв", examples)
        self.assertEqual(few[0], zero[0])
        self.assertEqual(few[-1], zero[-1])
        self.assertEqual(len(few), 12)
        self.assertEqual([m["content"] for m in few if m["role"] == "assistant"], list("12345"))
        self.assertEqual([m["role"] for m in few[1:-1]], ["user", "assistant"] * 5)
        with self.assertRaises(ValueError):
            few_shot_messages("Новый отзыв", examples[:4])


if __name__ == "__main__":
    unittest.main()
