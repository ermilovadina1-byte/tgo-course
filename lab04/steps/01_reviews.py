"""Первый шаг: загрузка отзывов и проверка разделения данных."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "lab04" / ".cache" / "huggingface"))

from lab04.data import REVISION, load_reviews, select_reviews, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = load_reviews(args.data)
    selection = select_reviews(data, seed=args.seed)
    revision = None if args.data or os.environ.get("LAB_DATA") else REVISION
    report = summarize(data, selection, seed=args.seed, source_revision=revision)
    report["fast_test_class_counts"] = summarize(
        data, select_reviews(data, seed=args.seed, fast=True), seed=args.seed,
    )["selected"]["test"]["class_counts"]
    out = ROOT / "lab04" / "artifacts" / "data_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Исходные разделы:", report["source_sizes"])
    for name, part in report["selected"].items():
        print(f"{name}: {part['n']} примеров; звезды -> число: {part['class_counts']}")
    print("Совпадения текста между выбранными разделами:", report["exact_text_overlap"])
    print("Примеры из train; метка в файле на единицу меньше числа звезд:")
    rows = data["train"].select(selection.train_indices)
    for label in (0, 4):
        example = min((r for r in rows if r["label"] == label), key=lambda r: len(r["text"]))
        print(f"  label={label}, звезд={label + 1}: {example['text']}")
    print("Пять примеров из validation, которые позже покажем модели в запросе:")
    for idx in selection.few_shot_indices:
        row = data["validation"][idx]
        print(f"  индекс {idx}, звезд={row['label'] + 1}: {row['text'][:120]}")
    print("Сводка сохранена:", out)


if __name__ == "__main__":
    main()
