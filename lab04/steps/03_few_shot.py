"""Сравнение двух запросов на тех же первых трех отзывах без обучения."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "lab04" / ".cache" / "huggingface"))

import torch
import transformers

from lab04.data import DATASET, REVISION, load_reviews, select_reviews
from lab04.inference import MODEL, MODEL_REVISION, load_qwen, predict_with_trace
from lab04.prompting import few_shot_messages, zero_shot_messages


def main():
    artifacts = ROOT / "lab04" / "artifacts"
    previous = json.loads((artifacts / "zero_shot_walkthrough.json").read_text())
    selection_record = json.loads((artifacts / "data_check.json").read_text())
    assert previous["model_revision"] == MODEL_REVISION
    assert previous["dataset_revision"] == REVISION
    data = load_reviews()
    selected = select_reviews(data, seed=42)
    assert selected.few_shot_indices == selection_record["indices"]["few_shot"]
    examples = [data["validation"][i] for i in selected.few_shot_indices]
    print("Пять примеров из validation, индексы:", selected.few_shot_indices, flush=True)
    model, tokenizer = load_qwen(local_only=True, threads=4)
    # Изменение тензоров на месте меняет их счетчики версий.
    before_versions = [(name, parameter._version) for name, parameter in model.named_parameters()]
    records = []
    for idx in range(3):
        row = data["test"][idx]
        old = previous["examples"][idx]
        assert old["test_idx"] == idx and old["review"] == row["text"]
        assert old["true_stars"] == row["label"] + 1
        traces = {}
        for mode, messages in (
            ("zero_shot", zero_shot_messages(row["text"])),
            ("few_shot", few_shot_messages(row["text"], examples)),
        ):
            traces[mode] = predict_with_trace(model, tokenizer, messages)
            traces[mode]["correct"] = traces[mode]["prediction"] == row["label"] + 1
        assert traces["zero_shot"]["input_ids"] == old["input_ids"]
        assert traces["zero_shot"]["output_ids"] == old["output_ids"]
        assert traces["zero_shot"]["messages"][0] == traces["few_shot"]["messages"][0]
        assert traces["zero_shot"]["messages"][-1] == traces["few_shot"]["messages"][-1]
        assert not any(ex["text"] == row["text"] for ex in examples)
        delta = traces["few_shot"]["prompt_tokens"] - traces["zero_shot"]["prompt_tokens"]
        records.append({"test_idx": idx, "review": row["text"], "true_stars": row["label"] + 1,
                        "added_prompt_tokens": delta, "modes": traces})
        print(f"test[{idx}], автор {row['label'] + 1}: "
              f"zero-shot={traces['zero_shot']['raw_answer']!r}, "
              f"few-shot={traces['few_shot']['raw_answer']!r}; "
              f"токены {traces['zero_shot']['prompt_tokens']} -> {traces['few_shot']['prompt_tokens']}", flush=True)
    unchanged = before_versions == [(name, p._version) for name, p in model.named_parameters()]
    assert unchanged and all(p.grad is None for p in model.parameters())
    report = {
        "model": MODEL, "model_revision": MODEL_REVISION,
        "dataset": DATASET, "dataset_revision": REVISION,
        "device": str(model.device), "dtype": str(model.dtype), "threads": 4,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "do_sample": False, "enable_thinking": False, "max_new_tokens": 3,
        "weights_unchanged": unchanged, "baseline_answers_reproduced": True,
        "n": 3, "selection": "Same first three test rows as the zero-shot walkthrough",
        "few_shot_examples": [{"validation_idx": i, "text": ex["text"], "stars": ex["label"] + 1}
                              for i, ex in zip(selected.few_shot_indices, examples)],
        "summary": {
            mode: {"correct": sum(r["modes"][mode]["correct"] for r in records),
                   "invalid": sum(r["modes"][mode]["prediction"] is None for r in records),
                   "mean_prompt_tokens": sum(r["modes"][mode]["prompt_tokens"] for r in records) / 3}
            for mode in ("zero_shot", "few_shot")
        },
        "examples": records,
    }
    out = artifacts / "few_shot_walkthrough.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    print("Веса не изменились; разбор сохранен:", out, flush=True)


if __name__ == "__main__":
    main()
