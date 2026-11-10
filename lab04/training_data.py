"""Подготовка ответа для LoRA; без загрузки модели и вычислений на GPU."""

from lab04.prompting import zero_shot_messages


def make_training_example(tokenizer, text, stars, max_len=256):
    if stars not in range(1, 6):
        raise ValueError("Правильная оценка должна быть от 1 до 5")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("У токенизатора нет маркера конца сообщения Qwen")
    answer_ids = tokenizer.encode(str(stars), add_special_tokens=False) + [end_id]

    def prompt_ids(review):
        rendered = tokenizer.apply_chat_template(
            zero_shot_messages(review), tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    original_ids = tokenizer.encode(text, add_special_tokens=False)
    prefix = prompt_ids(text)
    kept = len(original_ids)
    if len(prefix) + len(answer_ids) > max_len:
        budget = max_len - len(answer_ids) - len(prompt_ids(""))
        if budget < 1:
            raise ValueError("В заданную длину не помещаются инструкция и ответ")
        kept = min(kept, budget)
        while True:
            shortened = tokenizer.decode(original_ids[:kept], skip_special_tokens=False)
            prefix = prompt_ids(shortened)
            excess = len(prefix) + len(answer_ids) - max_len
            if excess <= 0:
                break
            kept -= max(1, excess)
            if kept < 1:
                raise ValueError("Не удалось сохранить отзыв в заданной длине")
    ids = prefix + answer_ids
    labels = [-100] * len(prefix) + answer_ids
    assert len(ids) == len(labels) <= max_len
    assert all(value == -100 for value in labels[:len(prefix)])
    assert labels[len(prefix):] == answer_ids and labels[-1] == end_id
    return {
        "input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels,
        "prompt_length": len(prefix), "answer_ids": answer_ids,
        "review_truncated": kept < len(original_ids),
    }


def pad_training_batch(examples, pad_id):
    """Возвращает списки, чтобы маски можно было проверить без torch."""
    if not examples:
        raise ValueError("Пустой батч")
    size = max(len(row["input_ids"]) for row in examples)
    batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in examples:
        padding = size - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
    return batch
