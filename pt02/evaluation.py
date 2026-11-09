"""Сравнение биграммы и трансформера по ошибке на одно слово текста."""

import hashlib
import math

import torch
from torch.nn import functional as F

from pt01.run import BigramModel


@torch.no_grad()
def token_nlls(model, ids: list[int], first_target: int, batch_size: int = 16):
    """Оцениваем каждый целевой токен один раз с максимумом доступного контекста."""
    if not 1 <= first_target < len(ids) or batch_size < 1:
        raise ValueError("Нужен непустой контекст, целевые токены и положительный батч")
    device = next(model.parameters()).device
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    context = model.config.context
    was_training = model.training
    model.eval()
    losses = [None] * len(ids)
    try:
        # Для начала текста один причинный проход дает все короткие префиксы.
        length = min(context, len(ids) - 1)
        logits = model(tokens[:length].unsqueeze(0))[0]
        initial = F.cross_entropy(logits, tokens[1:length + 1], reduction="none").tolist()
        for target, loss in enumerate(initial, start=1):
            if target >= first_target:
                losses[target] = loss

        # Затем у каждой цели ровно context предыдущих настоящих токенов.
        for start in range(max(context + 1, first_target), len(ids), batch_size):
            positions = range(start, min(start + batch_size, len(ids)))
            inputs = torch.stack([tokens[i - context:i] for i in positions])
            targets = tokens[start:start + len(inputs)]
            logits = model(inputs)[:, -1, :]
            values = F.cross_entropy(logits, targets, reduction="none").tolist()
            losses[start:start + len(values)] = values
    finally:
        model.train(was_training)
    if any(value is None or not math.isfinite(value) for value in losses[first_target:]):
        raise ValueError("Не все целевые токены получили конечную ошибку")
    return losses


def compare_models(model, tokenizer, train_words: list[str], val_words: list[str], word_limit: int = 2000):
    if word_limit < 2 or len(val_words) < word_limit:
        raise ValueError("Недостаточно слов отложенного текста для сравнения")
    words = val_words[:word_limit]
    bigram = BigramModel(alpha=0.01).fit(train_words)
    encoded_words = [tokenizer.encode_word(word) for word in words]
    ids = [token for encoded in encoded_words for token in encoded]
    first_target = len(encoded_words[0])
    token_losses = token_nlls(model, ids, first_target)
    bigram_losses = [-math.log(bigram.prob(previous, word)) for previous, word in zip(words, words[1:])]
    tiny_losses = []
    offset = first_target
    for encoded in encoded_words[1:]:
        tiny_losses.append(math.fsum(token_losses[offset:offset + len(encoded)]))
        offset += len(encoded)
    if offset != len(ids):
        raise ValueError("Границы слов не покрыли целевые BPE-токены")

    def metrics(losses, tokens):
        total = math.fsum(losses)
        return {"tokens": tokens, "nats_total": total,
                "nats_per_word": total / word_limit, "ppl_per_word": math.exp(total / word_limit)}

    unknown_indices = [i for i, word in enumerate(words[1:]) if word not in bigram.vocab]
    known_indices = [i for i, word in enumerate(words[1:]) if word in bigram.vocab]
    report = {
        "words": word_limit,
        "bigram": {**metrics(bigram_losses, word_limit - 1), "alpha": 0.01,
                   "vocab_size": len(bigram.vocab), "observed_pairs": len(bigram.pair_counts),
                   "unknown_target_words": len(unknown_indices)},
        "tiny": metrics(tiny_losses, len(ids) - first_target),
        "scored_words": word_limit - 1,
        "context_word": words[0], "context_bpe_tokens": first_target,
        "text_sha256": hashlib.sha256(" ".join(words).encode()).hexdigest(),
        "policy": {
            "text": "First word_limit words of the held-out split; split by words before BPE",
            "targets": "Both models score words 2..word_limit; the complete first word is context only",
            "denominator": "word_limit, including the initial context word, as in the assignment example",
            "tiny_context": f"Up to {model.config.context} previous true tokens; each target scored once",
            "probabilities": "Original model probabilities; no temperature or top-p filtering",
            "bigram_unknowns": "Unknown words are mapped to the existing pt01 <UNK> token",
        },
    }
    # Отдельно проверяем, сохраняется ли вывод без целей, замененных на <UNK>.
    if known_indices:
        report["known_target_words_only"] = {
            "scored_words": len(known_indices),
            "bigram_nats_per_scored_word": math.fsum(bigram_losses[i] for i in known_indices) / len(known_indices),
            "tiny_nats_per_scored_word": math.fsum(tiny_losses[i] for i in known_indices) / len(known_indices),
        }
    examples = []
    offset = first_target
    for i, encoded in enumerate(encoded_words[1:], start=1):
        if i <= 11:
            examples.append({
                "word_index": i, "word": words[i], "previous_word": words[i - 1],
                "bpe_tokens": [tokenizer.vocabulary[token] for token in encoded],
                "bigram_nats": bigram_losses[i - 1], "tiny_nats": tiny_losses[i - 1],
                "tiny_token_nats": token_losses[offset:offset + len(encoded)],
            })
        offset += len(encoded)
    return report, {"context_word": words[0], "examples": examples}
