"""Очная lab01: счётчики и генерация, без обучения нейросетей и BPE."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pt01.run import BigramModel, CORPUS_URL, compare_likelihood, load_corpus, split_words, tokenize_words

STARTS = ("князь", "она", "пьер", "наташа", "андрей")


def build_results():
    text = load_corpus()
    words = tokenize_words(text)
    train, validation = split_words(words)
    model = BigramModel(alpha=0.01).fit(train)
    generation = [{"start": start, "temperature": 1.0, "seed": 42,
                   "text": model.generate(start, n_words=30, temperature=1.0, seed=42)}
                  for start in STARTS]
    temperatures = [{"start": "князь", "temperature": t, "seed": 42,
                     "text": model.generate("князь", n_words=30, temperature=t, seed=42)}
                    for t in (0.5, 1.0, 2.0)]
    likelihood = compare_likelihood(model, validation, seed=42)
    unsmoothed = {}
    for name in ("sentence", "shuffled"):
        sequence = likelihood[name].split()
        pairs = list(zip(sequence, sequence[1:]))
        missing = sum(model.pair_counts[pair] == 0 for pair in pairs)
        value = None if missing else math.fsum(
            math.log(model.pair_counts[a, b] / model.context_counts[a]) for a, b in pairs)
        # null вместе с конечным флагом обозначает log(0), JSON Infinity не используем.
        unsmoothed[name] = {"unseen_pairs": missing, "finite": not missing, "logprob": value}
    perplexity = {}
    for alpha in (1.0, 0.1, 0.01):
        model.alpha = alpha
        perplexity[str(alpha)] = model.perplexity(validation)
    return {
        "corpus": "Война и мир", "source_url": CORPUS_URL,
        "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "words": len(words), "train_words": len(train), "val_words": len(validation),
        "vocab_size": len(set(words)), "train_vocab_size_with_unk": len(model.vocab),
        "top20": Counter(words).most_common(20), "perplexity": perplexity,
        "generation": generation, "temperature_generation": temperatures,
        "unsmoothed": unsmoothed, "likelihood": likelihood,
        "prediction_count": len(validation) - 1,
    }


def save_results(results, directory=HERE / "artifacts"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bigram.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    results = build_results()
    print(save_results(results, args.artifacts))
    print("Перплексия:", results["perplexity"])


if __name__ == "__main__":
    main()
