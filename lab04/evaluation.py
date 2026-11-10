"""Метрики оценок 1..5; некорректный ответ остается отдельной ошибкой."""

from collections import Counter


def score_ratings(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Нужны одинаковые непустые списки целей и предсказаний")
    if any(y not in range(1, 6) for y in labels):
        raise ValueError("Правильные оценки должны быть от 1 до 5")
    if any(p is not None and p not in range(1, 6) for p in predictions):
        raise ValueError("Предсказания должны быть от 1 до 5 или None")
    matrix = {str(y): {str(p): 0 for p in (1, 2, 3, 4, 5, "invalid")} for y in range(1, 6)}
    for y, p in zip(labels, predictions):
        matrix[str(y)][str(p) if p is not None else "invalid"] += 1
    per_class = {}
    for star in range(1, 6):
        tp = sum(y == star and p == star for y, p in zip(labels, predictions))
        fp = sum(y != star and p == star for y, p in zip(labels, predictions))
        fn = sum(y == star and p != star for y, p in zip(labels, predictions))
        per_class[str(star)] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    errors = [abs(y - p) for y, p in zip(labels, predictions) if p is not None]
    correct = sum(y == p for y, p in zip(labels, predictions))
    histogram = Counter(errors)
    return {
        "correct": correct, "accuracy": correct / len(labels),
        "f1_per_class": {star: row["f1"] for star, row in per_class.items()},
        "f1_macro": sum(row["f1"] for row in per_class.values()) / 5,
        "mae_stars": sum(errors) / len(errors) if errors else None,
        "invalid": len(labels) - len(errors), "invalid_rate": (len(labels) - len(errors)) / len(labels),
        "class_details": per_class, "confusion_matrix": matrix,
        "absolute_error_counts": {str(error): histogram[error] for error in range(5)},
        "prediction_counts": {str(star): predictions.count(star) for star in range(1, 6)},
    }
