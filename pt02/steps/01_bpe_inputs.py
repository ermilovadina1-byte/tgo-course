"""Загружаем BPE из pt01 и показываем сдвиг входов и целей на один токен."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pt01.run import tokenize_words
from pt02.data import load_bpe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="князь андрей почувствовал неудовольствие")
    parser.add_argument("--bpe", type=Path, default=ROOT / "pt01/artifacts/bpe.json")
    args = parser.parse_args()
    tokenizer, vocabulary = load_bpe(args.bpe)
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    words = tokenize_words(args.text)
    tokens = [token for word in words for token in tokenizer.encode_word(word)]
    if len(tokens) < 2:
        parser.error("Для пары вход-ответ нужны хотя бы два токена")
    ids = [token_to_id[token] for token in tokens]
    decoded = "".join(vocabulary[i] for i in ids).replace(tokenizer.END, " ").strip()
    assert decoded == " ".join(words)

    # Последний токен нужен как цель; перед ним заканчивается вход.
    inputs = ids[:-1]
    targets = ids[1:]
    assert len(inputs) == len(targets)
    assert inputs[1:] == targets[:-1]

    print(f"Загружено слияний: {len(tokenizer.merges)}; токенов в словаре: {len(vocabulary)}")
    print(f"Фраза: {' '.join(words)}")
    for word in words:
        print(f"  {word}: {tokenizer.encode_word(word)}")
    print(f"Токены: {tokens}")
    print(f"Номера: {ids}")
    print(f"Вход x: {inputs}")
    print(f"Цель y: {targets}")
    print("Позиция | доступный контекст | правильный следующий токен (номер)")
    for position, target in enumerate(targets):
        context = " ".join(tokens[:position + 1])
        print(f"{position} | {context} | {vocabulary[target]} ({target})")
    print(f"Обратное декодирование: {decoded}")


if __name__ == "__main__":
    main()
