"""Выбор следующих токенов: жадный, температурный и nucleus sampling."""

import math

import torch


def sampling_logits(logits: torch.Tensor, temperature: float, top_p: float | None = None):
    if logits.ndim != 1:
        raise ValueError("Ожидается один вектор логитов следующего токена")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Для случайного выбора температура должна быть положительной")
    if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("top_p должен быть в интервале (0, 1]")
    scaled = logits / temperature
    if top_p is None or top_p == 1:
        return scaled
    sorted_logits, order = scaled.sort(descending=True, stable=True)
    probabilities = sorted_logits.softmax(dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    # Включаем токен, пересекающий порог. Следующие токены уже не нужны.
    mass_before = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    keep = mass_before < top_p
    filtered = torch.full_like(scaled, float("-inf"))
    filtered[order[keep]] = sorted_logits[keep]
    return filtered


def first_repeat(ids: list[int], n: int = 4):
    """Начальная позиция первой повторной n-граммы среди новых токенов."""
    seen = set()
    for start in range(len(ids) - n + 1):
        group = tuple(ids[start:start + n])
        if group in seen:
            return start
        seen.add(group)
    return None


class TextGenerator:
    def __init__(self, model, tokenizer, seed: int = 42):
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int,
                 temperature: float = 1.0, top_p: float | None = None):
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens должен быть неотрицательным целым числом")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Температура должна быть конечной и неотрицательной")
        if top_p is not None and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p должен быть в интервале (0, 1]")
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("Затравка должна содержать токены")
        all_ids = list(prompt_ids)
        device = next(self.model.parameters()).device
        rng = torch.Generator(device="cpu").manual_seed(self.seed)
        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(max_new_tokens):
                window = torch.tensor([all_ids[-self.model.config.context:]], dtype=torch.long, device=device)
                logits = self.model(window)[0, -1]
                if temperature == 0:
                    next_id = logits.argmax().item()
                else:
                    probabilities = sampling_logits(logits, temperature, top_p).softmax(dim=-1).cpu()
                    next_id = torch.multinomial(probabilities, 1, generator=rng).item()
                all_ids.append(next_id)
        finally:
            self.model.train(was_training)
        new_ids = all_ids[len(prompt_ids):]
        return {"text": self.tokenizer.decode(all_ids), "ids": new_ids, "n_tokens": len(new_ids)}


def compare_generation(model, tokenizer, max_new_tokens: int = 80, seed: int = 42):
    generator = TextGenerator(model, tokenizer, seed)
    prompt = "князь андрей"
    modes = [("greedy", 0.0, None), ("t0.7", 0.7, None),
             ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)]
    runs = []
    for name, temperature, top_p in modes:
        result = generator.generate(prompt, max_new_tokens, temperature, top_p)
        row = {"name": name, "temperature": temperature, "top_p": top_p, **result}
        if name == "greedy":
            row["loop_start"] = first_repeat(row["ids"])
        runs.append(row)
    return {"model": "tiny", "prompt": prompt, "max_new_tokens": max_new_tokens,
            "seed": seed, "seed_policy": "Reset the same sampling seed for each mode",
            "ids_policy": "New tokens only; text includes the prompt",
            "runs": runs}
