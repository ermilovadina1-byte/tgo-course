"""Фиксированная подготовка отзывов для обучения, примеров и проверки."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random

from datasets import DatasetDict, load_dataset, load_from_disk

DATASET = "mteb/GeoreviewClassification"
REVISION = "fb29d6137c54550329411023aee45a8ff29f0c62"
HERE = Path(__file__).resolve().parent


def load_reviews(data_path=None):
    """LAB_DATA или --data задает папку, созданную save_to_disk."""
    source = data_path or os.environ.get("LAB_DATA")
    if source:
        data = load_from_disk(str(Path(source).expanduser()))
    else:
        data = load_dataset(
            DATASET, revision=REVISION,
            cache_dir=str(HERE / ".cache" / "datasets"),
        )
    if not isinstance(data, DatasetDict):
        raise ValueError("Нужны три раздела данных: train, validation, test")
    for split in ("train", "validation", "test"):
        if split not in data or not {"text", "label"} <= set(data[split].column_names):
            raise ValueError(f"В {split} нужны поля text и label")
        if any(label not in range(5) for label in data[split]["label"]):
            raise ValueError(f"В {split} метки должны быть целыми числами от 0 до 4")
        if any(not isinstance(t, str) or not t.strip() for t in data[split]["text"]):
            raise ValueError(f"В {split} обнаружен пустой или некорректный отзыв")
    return data


def balanced_indices(labels, per_class, seed):
    """Выбор без повторов; один проход по меткам и фиксированный генератор."""
    if per_class < 1:
        raise ValueError("Число примеров каждого класса должно быть положительным")
    groups = [[] for _ in range(5)]
    for idx, label in enumerate(labels):
        if label not in range(5):
            raise ValueError("Метка вне диапазона 0..4")
        groups[label].append(idx)
    rng = random.Random(seed)
    selected = []
    for label, indices in enumerate(groups):
        if len(indices) < per_class:
            raise ValueError(f"Для оценки {label + 1} нужно {per_class} примеров, есть {len(indices)}")
        selected.extend(rng.sample(indices, per_class))
    rng.shuffle(selected)
    return selected


@dataclass
class ReviewSelection:
    train_indices: list[int]
    few_shot_indices: list[int]
    test_indices: list[int]


def select_reviews(data, *, seed=42, fast=False, per_class=1000):
    train = balanced_indices(data["train"]["label"], per_class, seed)
    # Первое вхождение каждого класса: правило не зависит от качества ответов модели.
    validation_labels = list(data["validation"]["label"])
    few_shot = [validation_labels.index(label) for label in range(5)]
    n_test = 24 if fast else 1000
    if len(data["test"]) < n_test:
        raise ValueError(f"Для проверки нужны первые {n_test} отзывов")
    return ReviewSelection(train, few_shot, list(range(n_test)))


def star_counts(labels):
    counts = Counter(int(label) + 1 for label in labels)
    return {str(stars): counts[stars] for stars in range(1, 6)}


def selection_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        record = json.dumps([row["text"], int(row["label"])], ensure_ascii=False)
        digest.update((record + "\n").encode("utf-8"))
    return digest.hexdigest()


def summarize(data, selection, seed=42, source_revision=REVISION):
    selected = {
        "train": data["train"].select(selection.train_indices),
        "few_shot": data["validation"].select(selection.few_shot_indices),
        "test": data["test"].select(selection.test_indices),
    }
    text_sets = {name: set(rows["text"]) for name, rows in selected.items()}
    return {
        "dataset": DATASET,
        "source_revision": source_revision,
        "seed": seed,
        "label_mapping": "stars = label + 1",
        "source_sizes": {name: len(rows) for name, rows in data.items()},
        "selected": {
            name: {"n": len(rows), "class_counts": star_counts(rows["label"]),
                   "sha256": selection_digest(rows)}
            for name, rows in selected.items()
        },
        "indices": {
            "train": selection.train_indices,
            "few_shot": selection.few_shot_indices,
            "test": selection.test_indices,
        },
        "exact_text_overlap": {
            "train_test": len(text_sets["train"] & text_sets["test"]),
            "few_shot_test": len(text_sets["few_shot"] & text_sets["test"]),
            "train_few_shot": len(text_sets["train"] & text_sets["few_shot"]),
        },
    }
