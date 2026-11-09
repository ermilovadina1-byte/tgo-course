"""Учебный пример к pt01: частоты соседних слов и вероятности продолжения.

Запуск из корня проекта: /opt/homebrew/bin/python3.12 pt01/steps/01_counts.py
Это первый учебный шаг, а не готовый run.py для сдачи.
"""

import re
from collections import Counter


def main():
    text = "Кот спит. Кот ест. Кот спит."

    # Готовим слова по правилам задания. Пунктуация не сохраняется.
    words = re.findall(r"[а-я]+(?:-[а-я]+)*", text.lower().replace("ё", "е"))

    # zip соединяет каждое слово со следующим, Counter считает повторы пар.
    pairs = Counter(zip(words, words[1:]))

    # Выбираем только пары, в которых первое слово совпадает с контекстом.
    context = "кот"
    following = {right: count for (left, right), count in pairs.items() if left == context}
    total = sum(following.values())

    print("Слова:", words)
    print("Все соседние пары:", dict(pairs))
    print(f"После слова '{context}' найдено продолжений: {total}")
    for word, count in following.items():
        probability = count / total
        print(f"P({word} | {context}) = {count}/{total} = {probability:.3f}")


if __name__ == "__main__":
    main()
