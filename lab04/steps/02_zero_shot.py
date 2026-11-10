"""Разбор первых трех ответов Qwen: запрос, токены и жадная генерация."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "lab04" / ".cache" / "huggingface"))

import torch
import transformers

from lab04.data import DATASET, REVISION, load_reviews
from lab04.prompting import zero_shot_messages
from lab04.inference import MODEL, MODEL_REVISION, load_qwen, predict_with_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Число потоков должно быть положительным")
    print(f"Загрузка {MODEL}", flush=True)
    model, tokenizer = load_qwen(local_only=args.local_only, threads=args.threads)
    device, dtype = str(model.device), model.dtype
    data = load_reviews()
    records = []
    for idx in range(3):
        row = data["test"][idx]
        messages = zero_shot_messages(row["text"])
        trace = predict_with_trace(model, tokenizer, messages)
        record = {
            "test_idx": idx, "review": row["text"], **trace,
            "true_stars": int(row["label"]) + 1,
            "correct": trace["prediction"] == int(row["label"]) + 1,
        }
        records.append(record)
        print(f"test[{idx}]: вход {trace['prompt_tokens']} токенов; новые {trace['output_ids']}; "
              f"ответ {trace['raw_answer']!r}; оценка {trace['prediction']}; "
              f"правильная {record['true_stars']}; {trace['seconds']:.2f} с", flush=True)
        if idx == 0:
            print("Первый шаг, вероятности цифр:", trace["first_digit_probabilities"], flush=True)
            print("Самые вероятные токены:", trace["generation_steps"][0]["top5"], flush=True)
    report = {
        "model": MODEL, "model_revision": MODEL_REVISION, "dataset": DATASET,
        "dataset_revision": None if os.environ.get("LAB_DATA") else REVISION,
        "device": device, "dtype": str(dtype), "threads": args.threads,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "n_params": sum(p.numel() for p in model.parameters()),
        "hidden_size": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
        "enable_thinking": False, "do_sample": False, "max_new_tokens": 3,
        "selection": "First three test rows in original order; no tuning on their answers",
        "parser_policy": "Exactly one integer from 1 to 5; missing, fractional or ambiguous answers are invalid",
        "probability_policy": "Softmax over the full vocabulary, not normalized over five digits",
        "correct": sum(r["correct"] for r in records), "n": len(records), "examples": records,
    }
    out = ROOT / "lab04" / "artifacts" / "zero_shot_walkthrough.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Разбор сохранен:", out, flush=True)


if __name__ == "__main__":
    main()
