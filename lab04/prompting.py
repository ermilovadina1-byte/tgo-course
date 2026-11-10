"""Общий запрос оценки и однозначный разбор ответа модели."""

import re

SYSTEM_PROMPT = (
    "Определи, какую оценку от 1 до 5 поставил автор отзыва об организации. "
    "Шкала: 1 - очень плохо, 2 - плохо, 3 - удовлетворительно, "
    "4 - хорошо, 5 - отлично. Ответь одной цифрой от 1 до 5 без объяснений."
)


def review_message(text):
    return f"Отзыв:\n{text}\n\nКакую оценку поставил автор? Ответь одной цифрой от 1 до 5."


def zero_shot_messages(text):
    # Функция получает только текст: правильной оценки здесь нет.
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review_message(text)}]


def few_shot_messages(text, examples):
    """Пять фиксированных примеров из validation перед новым отзывом."""
    if len(examples) != 5 or sorted(row["label"] for row in examples) != list(range(5)):
        raise ValueError("Нужны пять примеров, по одному для каждой оценки")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for row in examples:
        messages.append({"role": "user", "content": review_message(row["text"])})
        messages.append({"role": "assistant", "content": str(row["label"] + 1)})
    messages.append({"role": "user", "content": review_message(text)})
    return messages


def parse_rating(answer):
    """Принимаем единственное отдельное целое число, если оно от 1 до 5."""
    numbers = re.findall(r"(?<![\w.,])[+-]?\d+(?:[.,]\d+)?(?!\w)", answer)
    if len(numbers) == 1 and numbers[0] in ("1", "2", "3", "4", "5"):
        return int(numbers[0])
    return None
