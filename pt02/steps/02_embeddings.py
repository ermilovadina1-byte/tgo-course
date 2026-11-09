"""Учебный пример выбора эмбеддингов и добавления позиций, без обучения."""

from pathlib import Path
import random
import runpy

HERE = Path(__file__).resolve().parent


def main():
    previous = runpy.run_path(str(HERE / "01_bpe_inputs.py"))
    tokenizer, vocabulary = previous["load_bpe"](
        HERE.parents[1] / "pt01/artifacts/bpe.json"
    )
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    words = ["князь", "андрей", "сказал"]
    tokens = [token for word in words for token in tokenizer.encode_word(word)]
    ids = [token_to_id[token] for token in tokens]
    inputs, targets = ids[:-1], ids[1:]

    # Четыре координаты и десятые доли выбраны для наглядности.
    # Это случайные начальные значения, не обученные признаки слов.
    width = 4
    rng = random.Random(42)
    token_embeddings = [
        [rng.randint(-5, 5) / 10 for _ in range(width)]
        for _ in vocabulary
    ]
    position_embeddings = [
        [rng.randint(-5, 5) / 10 for _ in range(width)]
        for _ in range(len(inputs))
    ]
    hidden = [
        [round(e + p, 1) for e, p in zip(token_embeddings[token_id], position_embeddings[pos])]
        for pos, token_id in enumerate(inputs)
    ]
    print("Учебный пример: случайные веса, обучение и внимание пока не выполняются.")
    print(f"Вход: {inputs}; цели: {targets}")
    print(f"Таблица токенов: {len(vocabulary)} строк x {width} числа")
    for pos, token_id in enumerate(inputs):
        print(f"Позиция {pos}, токен {vocabulary[token_id]}, номер {token_id}")
        print(f"  Строка таблицы токенов: {token_embeddings[token_id]}")
        print(f"  Строка таблицы позиций: {position_embeddings[pos]}")
        print(f"  Сумма:                 {hidden[pos]}")
    print(f"Вход в первый слой: {len(hidden)} позиции x {width} числа")
    print("При размерности 128: таблица токенов 1534 x 128, вход этого примера 2 x 128.")


if __name__ == "__main__":
    main()
