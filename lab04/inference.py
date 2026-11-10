"""Одинаковая генерация и разбор вероятностей для учебных экспериментов."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab04.prompting import parse_rating

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def load_qwen(*, local_only=True, threads=4, dtype=None):
    if threads < 1:
        raise ValueError("Число потоков должно быть положительным")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = dtype or (torch.float32 if device == "cpu" else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=dtype, local_files_only=local_only,
    ).to(device).eval()
    return model, tokenizer


def predict_with_trace(model, tokenizer, messages, *, collect_trace=True):
    """На входе только сообщения; правильная оценка не участвует в вычислении."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0].tolist()
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            **inputs, do_sample=False, max_new_tokens=3, use_cache=True,
            temperature=None, top_p=None, top_k=None,
            return_dict_in_generate=True, output_logits=collect_trace,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        )
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elif model.device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    new_ids = result.sequences[0, len(input_ids):].tolist()
    raw_answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    if not collect_trace:
        return {"prompt_tokens": len(input_ids), "output_ids": new_ids,
                "raw_answer": raw_answer, "prediction": parse_rating(raw_answer), "seconds": seconds}
    steps = []
    for token_id, logits in zip(new_ids, result.logits):
        probabilities = logits[0].float().softmax(dim=-1)
        values, ids = probabilities.topk(5)
        assert token_id == int(logits[0].argmax()), "Жадный выбор не совпал с максимумом логитов"
        steps.append({
            "chosen_id": token_id, "chosen_text": tokenizer.decode([token_id]),
            "chosen_probability": float(probabilities[token_id]),
            "top5": [{"id": i, "text": tokenizer.decode([i]), "probability": p}
                     for i, p in zip(ids.tolist(), values.tolist())],
        })
    first_probs = result.logits[0][0].float().softmax(dim=-1)
    digit_probs = {}
    for digit in "12345":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        assert len(ids) == 1, "Для таблицы вероятностей цифра должна быть одним токеном"
        digit_probs[digit] = {"id": ids[0], "probability": float(first_probs[ids[0]])}
    return {
        "messages": messages, "rendered_prompt": rendered, "input_ids": input_ids,
        "prompt_tokens": len(input_ids),
        "token_preview": [{"id": i, "text": tokenizer.decode([i])} for i in input_ids[:35]],
        "output_ids": new_ids, "raw_answer": raw_answer, "prediction": parse_rating(raw_answer),
        "generation_steps": steps, "first_digit_probabilities": digit_probs, "seconds": seconds,
    }
