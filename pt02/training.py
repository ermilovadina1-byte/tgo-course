"""Окна текста, оценка без обновления весов и образцы по ходу обучения."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from pt02.generation import TextGenerator


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sample_windows(tokens: torch.Tensor, batch: int, context: int,
                   generator: torch.Generator):
    if tokens.ndim != 1 or len(tokens) <= context:
        raise ValueError("Поток должен содержать хотя бы context + 1 токенов")
    starts = torch.randint(len(tokens) - context, (batch,), generator=generator)
    windows = tokens[starts[:, None] + torch.arange(context + 1)]
    return windows[:, :-1], windows[:, 1:], starts


@torch.no_grad()
def evaluate_loss(model, x, y, batch_size: int = 4) -> float:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(x), batch_size):
            inputs, targets = x[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
            logits = model(inputs)
            total_loss += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
            ).item()
            total_tokens += targets.numel()
    finally:
        model.train(was_training)
    return total_loss / total_tokens


@torch.no_grad()
def sample_text(model, tokenizer, prompt: str, max_new_tokens: int = 60,
                temperature: float = 0.8, seed: int = 42) -> str:
    if temperature <= 0:
        raise ValueError("Для образцов обучения нужна положительная температура")
    return TextGenerator(model, tokenizer, seed).generate(
        prompt, max_new_tokens, temperature
    )["text"]


def plot_curve(rows: list[dict], path: Path):
    import os
    import tempfile
    cache = Path(tempfile.gettempdir()) / "mtusi-pt02-plot-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot([r["step"] for r in rows], [r["train_loss"] for r in rows],
            label="Обучение: средняя ошибка за интервал", color="#555555", linestyle="--")
    ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows],
            label="Проверка: 20 фиксированных окон", color="#b63c67", linewidth=2)
    ax.set(xlabel="Шаг обновления весов", ylabel="Ошибка, нат на токен",
           title="Обучение трансформера на «Войне и мире»")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.text(0.5, -0.03, "На шаге 0 обе ошибки измерены на фиксированных окнах до обучения.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
