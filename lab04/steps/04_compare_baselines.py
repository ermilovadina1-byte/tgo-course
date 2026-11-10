"""Полная оценка zero-shot и few-shot до обучения адаптера."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "lab04" / ".cache" / "huggingface"))

import torch
import transformers

from lab04.data import DATASET, REVISION, load_reviews, select_reviews, selection_digest, star_counts
from lab04.evaluation import score_ratings
from lab04.inference import MODEL, MODEL_REVISION, load_qwen, predict_with_trace
from lab04.prompting import few_shot_messages, zero_shot_messages


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "lab04/artifacts/baseline")
    args = parser.parse_args()
    out = args.artifacts.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = load_reviews()
    selected = select_reviews(data, seed=42, fast=args.fast)
    test = data["test"].select(selected.test_indices)
    examples = [data["validation"][i] for i in selected.few_shot_indices]
    source_check = json.loads((ROOT / "lab04/artifacts/data_check.json").read_text())
    assert selected.few_shot_indices == source_check["indices"]["few_shot"]
    if not args.fast:
        assert selection_digest(test) == source_check["selected"]["test"]["sha256"]
    assert not set(row["text"] for row in examples) & set(test["text"])
    print(f"Загрузка модели; проверочных отзывов {len(test)}", flush=True)
    model, tokenizer = load_qwen(local_only=True, threads=4, dtype=torch.float32)
    versions = [(name, parameter._version) for name, parameter in model.named_parameters()]
    print(f"Устройство {model.device}, тип чисел {model.dtype}", flush=True)
    builders = {"zero_shot": zero_shot_messages,
                "few_shot": lambda text: few_shot_messages(text, examples)}
    previous = json.loads((ROOT / "lab04/artifacts/few_shot_walkthrough.json").read_text())
    # Проверка переноса CPU -> GPU заодно прогревает оба вида запроса.
    preflight = []
    for row in previous["examples"]:
        for mode, build in builders.items():
            trace = predict_with_trace(model, tokenizer, build(row["review"]))
            reference = row["modes"][mode]
            assert trace["input_ids"] == reference["input_ids"]
            assert trace["output_ids"] == reference["output_ids"], (row["test_idx"], mode)
            delta = max(abs(trace["first_digit_probabilities"][star]["probability"] -
                            reference["first_digit_probabilities"][star]["probability"])
                        for star in "12345")
            preflight.append({"idx": row["test_idx"], "mode": mode,
                              "output_ids_match": True, "max_digit_probability_delta": delta})
    print("Проверка первых трех отзывов: прежние ответы воспроизведены в обоих режимах", flush=True)
    labels = [int(label) + 1 for label in test["label"]]
    records = [{"idx": idx, "label": label} for idx, label in zip(selected.test_indices, labels)]
    metrics = {
        "model": MODEL, "model_revision": MODEL_REVISION,
        "dataset": DATASET, "dataset_revision": None if os.environ.get("LAB_DATA") else REVISION,
        "test_sha256": selection_digest(test), "n_test": len(test), "fast": args.fast,
        "class_counts": star_counts(test["label"]), "device": str(model.device),
        "dtype": str(model.dtype), "threads": 4,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "do_sample": False, "enable_thinking": False, "max_new_tokens": 3,
        "few_shot_indices": selected.few_shot_indices,
        "few_shot_examples_sha256": selection_digest(data["validation"].select(selected.few_shot_indices)),
        "preflight": preflight,
        "status": "running", "modes": {},
        "policy": {
            "test": "First 24 or 1000 test rows in their original order; no prompt tuning",
            "invalid": "Wrong for accuracy and F1; excluded from MAE and counted separately",
            "f1_macro": "Unweighted mean of F1 for all five classes, including zero scores",
            "seconds": "Wall time per request, including tokenization, generation, device synchronization and parsing; excludes model loading and six warmup requests",
            "cache": "KV cache within each generation; no shared prefix cache between reviews",
        },
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(), ROOT / "lab04/inference.py",
                                    ROOT / "lab04/prompting.py", ROOT / "lab04/evaluation.py"]},
    }
    save_json(out / "metrics.json", metrics)
    with (out / "requests.jsonl").open("w", encoding="utf-8") as journal:
        for mode, build in builders.items():
            mode_start = time.perf_counter()
            for idx, row in enumerate(test):
                start = time.perf_counter()
                prediction = predict_with_trace(model, tokenizer, build(row["text"]), collect_trace=False)
                seconds = time.perf_counter() - start
                records[idx].update({mode: prediction["prediction"], f"raw_{mode}": prediction["raw_answer"],
                                     f"tokens_{mode}": prediction["prompt_tokens"], f"seconds_{mode}": seconds})
                journal.write(json.dumps({"idx": idx, "mode": mode, "label": labels[idx],
                                          **prediction, "request_seconds": seconds}, ensure_ascii=False) + "\n")
                journal.flush()
                if (idx + 1) % 50 == 0 or idx + 1 == len(test):
                    correct = sum(r[mode] == r["label"] for r in records[:idx + 1])
                    progress = {"mode": mode, "completed": idx + 1, "total": len(test),
                                "correct": correct, "accuracy_so_far": correct / (idx + 1),
                                "mode_elapsed_seconds": time.perf_counter() - mode_start}
                    save_json(out / "progress.json", progress)
                    print(f"{mode}: {idx + 1}/{len(test)}, совпало {correct}, "
                          f"точность {correct / (idx + 1):.3f}, {time.perf_counter() - mode_start:.1f} с", flush=True)
            result = score_ratings(labels, [r[mode] for r in records])
            result["prompt_tokens"] = sum(r[f"tokens_{mode}"] for r in records) / len(test)
            result["sec_per_example"] = sum(r[f"seconds_{mode}"] for r in records) / len(test)
            metrics["modes"][mode] = result
            save_json(out / "metrics.json", metrics)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    assert versions == [(name, p._version) for name, p in model.named_parameters()]
    assert all(p.grad is None for p in model.parameters())
    metrics["weights_unchanged"] = True
    metrics["status"] = "completed"
    metrics["process_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    metrics["paired_comparison"] = {
        "same_answer": sum(r["zero_shot"] == r["few_shot"] for r in records),
        "both_correct": sum(r["zero_shot"] == r["label"] == r["few_shot"] for r in records),
        "fixed_by_few_shot": sum(r["zero_shot"] != r["label"] == r["few_shot"] for r in records),
        "broken_by_few_shot": sum(r["zero_shot"] == r["label"] != r["few_shot"] for r in records),
    }
    save_json(out / "metrics.json", metrics)
    print("Сравнение завершено:", out, flush=True)


if __name__ == "__main__":
    main()
