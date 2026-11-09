"""Лабораторная pt01. Биграммная языковая модель и собственный BPE.

Запуск: /opt/homebrew/bin/python3.12 pt01/run.py
Быстрый прогон: /opt/homebrew/bin/python3.12 pt01/run.py --fast
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.txt"
ARTIFACTS = HERE / "artifacts"
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/"
    "word2vec-russian-novels/master/books_before/WarAndPeace.txt"
)


def load_corpus() -> str:
    """Скачиваем книгу один раз, следующие запуски читают локальную копию."""
    if CORPUS.exists():
        return CORPUS.read_text(encoding="utf-8")

    print("Скачиваем корпус: Война и мир", flush=True)
    # Таймаут 60 секунд задан нами для сетевого чтения, это не лимит сервера.
    with urlopen(CORPUS_URL, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    if not text.strip():
        raise ValueError("Скачанный корпус пуст")

    # Незавершенная запись не должна выглядеть как готовый корпус.
    temporary = CORPUS.with_suffix(".txt.part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(CORPUS)
    return text


def tokenize_words(text: str) -> list[str]:
    """Выделяем слова строго по правилу из задания, это еще не BPE."""
    normalized = text.lower().replace("ё", "е")
    return re.findall(r"[а-я]+(?:-[а-я]+)*", normalized)


def split_words(words: list[str]) -> tuple[list[str], list[str]]:
    """Последние примерно 10% слов оставляем для проверки модели."""
    if len(words) < 10:
        raise ValueError("Для разделения корпуса нужно хотя бы 10 слов")
    split_at = len(words) * 9 // 10
    return words[:split_at], words[split_at:]


def temperature_distribution(probabilities: list[float], temperature: float) -> list[float]:
    """P_T(w) = P(w) ** (1 / T), затем нормировка суммы к единице."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Температура должна быть конечной и положительной")
    if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Нужен непустой список вероятностей от 0 до 1")
    if not any(probabilities):
        raise ValueError("Хотя бы одна вероятность должна быть положительной")
    # Вычисляем степени через логарифмы и вычитаем максимум для устойчивости.
    # Общий множитель сократится при нормировке, результат не меняется.
    log_probs = [math.log(p) if p > 0 else -math.inf for p in probabilities]
    maximum = max(log_probs)
    weights = [math.exp((value - maximum) / temperature) for value in log_probs]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


class BigramModel:
    """Счетчики соседних слов со сглаживанием Лапласа."""

    UNK = "<UNK>"

    def __init__(self, alpha: float = 0.01):
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha должна быть конечной и положительной")
        self.alpha = alpha
        self.vocab = set()

    def fit(self, words: list[str]) -> "BigramModel":
        if len(words) < 2:
            raise ValueError("Для обучения нужна хотя бы одна пара слов")
        if self.UNK in words:
            raise ValueError("Маркер <UNK> зарезервирован для незнакомых слов")
        self.word_counts = Counter(words)
        self.vocab = set(words) | {self.UNK}
        self.pair_counts = Counter(zip(words, words[1:]))
        # Последнее слово не имеет продолжения, поэтому не входит в знаменатель.
        self.context_counts = Counter(words[:-1])
        return self

    def prob(self, previous: str, word: str) -> float:
        if not self.vocab:
            raise ValueError("Сначала вызовите fit")
        previous = previous if previous in self.vocab else self.UNK
        word = word if word in self.vocab else self.UNK
        pair_count = self.pair_counts[previous, word]
        context_count = self.context_counts[previous]
        # Прибавляем alpha каждому возможному продолжению из словаря.
        return (pair_count + self.alpha) / (
            context_count + self.alpha * len(self.vocab)
        )

    def logprob(self, words: list[str]) -> float:
        if not self.vocab:
            raise ValueError("Сначала вызовите fit")
        return math.fsum(
            math.log(self.prob(previous, word))
            for previous, word in zip(words, words[1:])
        )

    def perplexity(self, words: list[str]) -> float:
        if len(words) < 2:
            raise ValueError("Для оценки нужна хотя бы одна пара слов")
        # Первое слово служит контекстом: оцененных продолжений на одно меньше.
        predictions = len(words) - 1
        return math.exp(-self.logprob(words) / predictions)


    def generate(self, start: str, n_words: int = 30, temperature: float = 1.0, seed: int = 42) -> str:
        """Генерируем n_words элементов вместе со стартовым словом."""
        if not self.vocab:
            raise ValueError("Сначала вызовите fit")
        if not isinstance(n_words, int) or isinstance(n_words, bool) or n_words < 1:
            raise ValueError("n_words должно быть положительным целым числом")
        if not isinstance(start, str) or not start or len(start.split()) != 1 or start != start.strip():
            raise ValueError("Начало должно состоять ровно из одного слова")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Температура должна быть конечной и положительной")

        # Порядок set меняется между процессами, поэтому сортируем словарь.
        vocabulary = sorted(self.vocab)
        rng = random.Random(seed)
        words = [start]
        for _ in range(n_words - 1):
            previous = words[-1]
            probabilities = [self.prob(previous, word) for word in vocabulary]
            adjusted = temperature_distribution(probabilities, temperature)
            next_word = rng.choices(vocabulary, weights=adjusted, k=1)[0]
            words.append(next_word)
        return " ".join(words)


def compare_likelihood(model: BigramModel, val_words: list[str], seed: int = 42) -> dict:
    """Сравниваем первые 12 проверочных слов и одну фиксированную перестановку."""
    if len(val_words) < 12:
        raise ValueError("Для сравнения нужны хотя бы 12 проверочных слов")
    sentence = val_words[:12]
    shuffled = sentence.copy()
    random.Random(seed).shuffle(shuffled)

    def trace(words: list[str]) -> list[dict]:
        return [
            {
                "previous": previous,
                "word": word,
                "train_pair_count": model.pair_counts[previous, word],
                "prob": model.prob(previous, word),
                "logprob": math.log(model.prob(previous, word)),
            }
            for previous, word in zip(words, words[1:])
        ]

    sentence_trace = trace(sentence)
    shuffled_trace = trace(shuffled)
    return {
        "sentence": " ".join(sentence),
        "shuffled": " ".join(shuffled),
        "alpha": model.alpha,
        "seed": seed,
        "predictions": len(sentence) - 1,
        "logprob_sentence": model.logprob(sentence),
        "logprob_shuffled": model.logprob(shuffled),
        "unseen_pairs_in_sentence": sum(row["train_pair_count"] == 0 for row in sentence_trace),
        "unseen_pairs_in_shuffled": sum(row["train_pair_count"] == 0 for row in shuffled_trace),
        "sentence_trace": sentence_trace,
        "shuffled_trace": shuffled_trace,
        "scoring_policy": "Sum of 11 conditional natural log probabilities; first word is not scored.",
    }


def evaluate_bigrams(train_words: list[str], val_words: list[str]) -> dict:
    model = BigramModel().fit(train_words)
    scores = {}
    mean_losses = {}
    for alpha in (1, 0.1, 0.01):
        model.alpha = alpha
        mean_loss = -model.logprob(val_words) / (len(val_words) - 1)
        scores[str(alpha)] = math.exp(mean_loss)
        mean_losses[str(alpha)] = mean_loss

    model.alpha = 0.01
    continuations = {}
    for context in ("князь", "она"):
        observed = Counter({
            word: count
            for (previous, word), count in model.pair_counts.items()
            if previous == context
        })
        continuations[context] = {
            "context_count": model.context_counts[context],
            "top5": [
                {"word": word, "count": count, "prob_alpha_0.01": model.prob(context, word)}
                for word, count in observed.most_common(5)
            ],
        }

    trace = [
        {
            "previous": previous,
            "word": word,
            "pair_count": model.pair_counts[previous, word],
            "prob": model.prob(previous, word),
            "negative_logprob": -math.log(model.prob(previous, word)),
        }
        for previous, word in zip(val_words[:9], val_words[1:9])
    ]
    unknown_words = [word for word in val_words if word not in model.vocab]
    generation = [
        {
            "temperature": temperature,
            "start": start,
            "text": model.generate(start, n_words=30, temperature=temperature, seed=42),
        }
        for temperature in (0.5, 1.0, 2.0)
        for start in ("князь", "она")
    ]
    vocabulary = sorted(model.vocab)
    base = [model.prob("князь", word) for word in vocabulary]
    temperature_effects = []
    for temperature in (0.5, 1.0, 2.0):
        distribution = temperature_distribution(base, temperature)
        temperature_effects.append({
            "temperature": temperature,
            "context": "князь",
            "p_andrey": distribution[vocabulary.index("андрей")],
            "unseen_pair_mass": math.fsum(
                p for word, p in zip(vocabulary, distribution)
                if model.pair_counts["князь", word] == 0
            ),
        })
    return {
        "stage": "bigram_complete",
        "corpus": CORPUS_URL,
        "train_words": len(train_words),
        "val_words": len(val_words),
        "train_pairs": sum(model.pair_counts.values()),
        "distinct_train_pairs": len(model.pair_counts),
        "train_vocab_size": len(model.word_counts),
        "vocab_size": len(model.vocab),
        "vocab_scope": "training_words_plus_UNK",
        "unknown_policy": "All words absent from training map to one <UNK> category.",
        "val_unknown_occurrences": len(unknown_words),
        "val_unknown_types": len(set(unknown_words)),
        "val_predictions": len(val_words) - 1,
        "perplexity": scores,
        "mean_negative_logprob": mean_losses,
        "continuations": continuations,
        "validation_trace_alpha_0.01": trace,
        "generation": generation,
        "generation_settings": {
            "alpha": 0.01,
            "seed": 42,
            "seed_policy": "Reset the same seed for each sample; vocabulary sorted.",
            "words_including_start": 30,
            "vocabulary_policy": "Full model vocabulary including <UNK>; no filtering.",
        },
        "temperature_effects": temperature_effects,
        "likelihood": compare_likelihood(model, val_words, seed=42),
    }


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Заменяем все непересекающиеся вхождения пары слева направо."""
    merged = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    """BPE внутри слов: частые пары сливаются, порядок правил сохраняется."""

    END = "</w>"

    def __init__(self):
        self.merges = []
        self.ranks = {}
        self.fitted = False

    def fit(self, word_counts: Counter, num_merges: int = 1500,
            min_frequency: int = 5, verbose: bool = False) -> "BPETokenizer":
        if not isinstance(num_merges, int) or isinstance(num_merges, bool) or num_merges < 0:
            raise ValueError("num_merges должно быть неотрицательным целым числом")
        if not isinstance(min_frequency, int) or isinstance(min_frequency, bool) or min_frequency < 1:
            raise ValueError("min_frequency должно быть положительным целым числом")
        if not word_counts or any(
            not isinstance(word, str) or not word or self.END in word
            or not isinstance(count, int) or isinstance(count, bool) or count < 1
            for word, count in word_counts.items()
        ):
            raise ValueError("Нужны непустые слова с положительными целыми частотами")
        selected = sorted((word, count) for word, count in word_counts.items() if count >= min_frequency)
        if not selected:
            raise ValueError("Нет слов с частотой не ниже порога")
        sequences = [tuple(word) + (self.END,) for word, _ in selected]
        frequencies = [count for _, count in selected]
        self.merges = []
        self.ranks = {}
        self.merge_frequencies = []
        self.fitted = False
        self.symbols = {char for word in word_counts for char in word} | {self.END}
        self.selected_types = len(selected)
        self.selected_occurrences = sum(frequencies)

        for step in range(num_merges):
            pair_counts = Counter()
            for symbols, frequency in zip(sequences, frequencies):
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break
            # При равной частоте выбираем лексикографически: результат воспроизводим.
            best = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            self.merges.append(best)
            self.merge_frequencies.append(pair_counts[best])
            self.symbols.add(best[0] + best[1])
            sequences = [
                merge_pair(symbols, best) if best in zip(symbols, symbols[1:]) else symbols
                for symbols in sequences
            ]
            if verbose and (step < 5 or (step + 1) % 100 == 0):
                print(f"BPE {step + 1}/{num_merges}: {best!r}, частота {pair_counts[best]}", flush=True)

        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.fitted = True
        return self

    def encode_word(self, word: str) -> list[str]:
        """Применяем выученные правила по рангу, как требует контракт курса."""
        if not self.fitted:
            raise ValueError("Сначала вызовите fit")
        if not word or self.END in word:
            raise ValueError("Нужно непустое слово без служебного маркера")
        symbols = list(word) + [self.END]
        while len(symbols) > 1:
            candidates = [
                (self.ranks[pair], i)
                for i, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self.ranks
            ]
            if not candidates:
                break
            _, i = min(candidates)
            symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
        return symbols


def evaluate_bpe(word_counts: Counter, num_merges: int, verbose: bool = True) -> dict:
    tokenizer = BPETokenizer().fit(word_counts, num_merges, min_frequency=5, verbose=verbose)
    if len(tokenizer.merges) != num_merges:
        raise ValueError("Для требуемого числа слияний недостаточно пар")
    # Считаем каждое различное слово один раз, затем учитываем его частоту.
    encoded = {word: tokenizer.encode_word(word) for word in word_counts}
    n_words = sum(word_counts.values())
    n_chars = sum(len(word) * count for word, count in word_counts.items())
    n_tokens = sum(len(encoded[word]) * count for word, count in word_counts.items())
    return {
        "num_merges": len(tokenizer.merges),
        "vocab_size": len(tokenizer.symbols),
        "merges": tokenizer.merges,
        "stats": {
            "chars": n_chars,
            "words": n_words,
            "tokens_bpe": n_tokens,
            "tokens_per_word": n_tokens / n_words,
            "compression_vs_chars": n_chars / n_tokens,
        },
        "examples": {word: encoded[word] for word in (
            "наполеон", "андрей", "сказала", "неудовольствие", "лошадь"
        )},
        "training": {
            "scope": "whole_corpus_words",
            "min_frequency": 5,
            "selected_word_types": tokenizer.selected_types,
            "selected_word_occurrences": tokenizer.selected_occurrences,
            "tie_break": "lexicographic_pair_order",
            "first10_merge_frequencies": tokenizer.merge_frequencies[:10],
            "char_count_policy": "Characters in extracted words, including hyphens; no spaces or end markers.",
            "token_count_policy": "All resulting tokens, including standalone or merged word-end markers.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="pt01: биграммы и BPE")
    parser.add_argument("--fast", action="store_true", help="200 слияний BPE вместо 1500")
    args = parser.parse_args()
    started = time.perf_counter()
    text = load_corpus()
    words = tokenize_words(text)
    train_words, val_words = split_words(words)

    # Это описательная статистика всего корпуса, не обучение модели.
    counts = Counter(words)
    stats = {
        "stage": "corpus_preparation",
        "corpus": "WarAndPeace",
        "source_url": CORPUS_URL,
        "words": len(words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "vocab_size": len(counts),
        "vocab_scope": "whole_corpus",
        "top20": counts.most_common(20),
        "train_first12": train_words[:12],
        "val_first12": val_words[:12],
    }
    ARTIFACTS.mkdir(exist_ok=True)
    output = ARTIFACTS / "corpus_stats.json"
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Всего слов: {len(words):,}")
    print(f"Разных слов во всем корпусе: {len(counts):,}")
    print(f"Для обучения: {len(train_words):,}")
    print(f"Для проверки: {len(val_words):,} ({len(val_words) / len(words):.2%})")
    print("20 самых частых слов во всем корпусе:")
    for word, count in counts.most_common(20):
        print(f"  {word}: {count}")
    print("Первые 12 слов проверочной части:", " ".join(val_words[:12]))
    print(f"Статистика сохранена: {output}")
    result = evaluate_bigrams(train_words, val_words)
    result["words"] = len(words)
    result["top20"] = stats["top20"]
    bigram_output = ARTIFACTS / "bigram.json"
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    bigram_output.write_text(serialized, encoding="utf-8")
    # Сохраняем актуальной ссылку на промежуточный файл из предыдущих шагов.
    (ARTIFACTS / "bigram_progress.json").write_text(serialized, encoding="utf-8")
    likelihood_output = ARTIFACTS / "likelihood.json"
    likelihood_output.write_text(
        json.dumps(result["likelihood"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    generation_output = ARTIFACTS / "generation.json"
    generation_output.write_text(json.dumps({
        "settings": result["generation_settings"],
        "generation": result["generation"],
        "temperature_effects": result["temperature_effects"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Обучающих пар: {result['train_pairs']:,}")
    print(f"Разных обучающих пар: {result['distinct_train_pairs']:,}")
    print(f"Словарь модели: {result['train_vocab_size']:,} слов + <UNK>")
    print(f"Незнакомых слов в проверке: {result['val_unknown_occurrences']:,} вхождений")
    for context, details in result["continuations"].items():
        print(f"После '{context}' наблюдалось {details['context_count']} продолжений:")
        for row in details["top5"]:
            print(f"  {row['word']}: {row['count']} раз, P = {row['prob_alpha_0.01']:.2%} при alpha=0.01")
    print(f"Перплексия на {result['val_predictions']:,} проверочных парах:")
    for alpha, score in result["perplexity"].items():
        print(f"  alpha={alpha}: {score:.2f}")
    print(f"Результат сохранен: {bigram_output}")
    for sample in result["generation"]:
        print(f"T={sample['temperature']}, начало '{sample['start']}': {sample['text']}")
    print(f"Генерация сохранена: {generation_output}")
    likelihood = result["likelihood"]
    print("Исходный фрагмент:", likelihood["sentence"])
    print("Перемешанный фрагмент:", likelihood["shuffled"])
    print(f"log P исходного: {likelihood['logprob_sentence']:.6f}")
    print(f"log P перемешанного: {likelihood['logprob_shuffled']:.6f}")
    print(f"Ненаблюдавшихся пар: исходный {likelihood['unseen_pairs_in_sentence']}, перемешанный {likelihood['unseen_pairs_in_shuffled']}")
    print(f"Подробное сравнение: {likelihood_output}")
    print("Обучаем BPE на словах всего корпуса с частотой не ниже 5", flush=True)
    bpe = evaluate_bpe(counts, num_merges=200 if args.fast else 1500)
    bpe_output = ARTIFACTS / "bpe.json"
    bpe_output.write_text(json.dumps(bpe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"BPE: {bpe['num_merges']} слияний, словарь {bpe['vocab_size']} токенов")
    print(f"Символов: {bpe['stats']['chars']}; токенов BPE: {bpe['stats']['tokens_bpe']}")
    print(f"Токенов на слово: {bpe['stats']['tokens_per_word']:.4f}; сжатие: {bpe['stats']['compression_vs_chars']:.4f}")
    for word, tokens in bpe["examples"].items():
        print(f"{word}: {tokens}")
    print(f"BPE сохранен: {bpe_output}")
    print(f"Полный скрипт выполнен за {time.perf_counter() - started:.2f} с")


if __name__ == "__main__":
    main()
