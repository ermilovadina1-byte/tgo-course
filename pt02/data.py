"""Сохраненный BPE и перевод нормализованных слов в номера токенов."""

from functools import lru_cache
import json
from pathlib import Path

from pt01.run import BPETokenizer, tokenize_words


def load_bpe(path: Path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = BPETokenizer()
    tokenizer.merges = [tuple(pair) for pair in saved["merges"]]
    tokenizer.ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
    tokenizer.fitted = True
    symbols = set("абвгдежзийклмнопрстуфхцчшщъыьэюя-") | {tokenizer.END}
    for left, right in tokenizer.merges:
        symbols.update((left, right, left + right))
    vocabulary = sorted(symbols)
    if len(vocabulary) != saved["vocab_size"]:
        raise ValueError("Размер восстановленного словаря BPE не совпал с сохраненным")
    for word, expected in saved["examples"].items():
        if tokenizer.encode_word(word) != expected:
            raise ValueError(f"Сохраненные правила BPE не воспроизвели пример {word}")
    return tokenizer, vocabulary


class BPETokens:
    def __init__(self, path: Path):
        self.bpe, self.vocabulary = load_bpe(path)
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}

    @lru_cache(maxsize=None)
    def encode_word(self, word: str) -> tuple[int, ...]:
        return tuple(self.token_to_id[token] for token in self.bpe.encode_word(word))

    def encode_words(self, words: list[str]) -> list[int]:
        return [token for word in words for token in self.encode_word(word)]

    def encode(self, text: str) -> list[int]:
        return self.encode_words(tokenize_words(text))

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocabulary[i] for i in ids).replace(self.bpe.END, " ").strip()
