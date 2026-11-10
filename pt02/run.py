"""pt02: обучение своего трансформера, образцы и режимы генерации."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import torch
from torch.nn import functional as F

from pt01.run import load_corpus, split_words, tokenize_words
from pt02.data import BPETokens
from pt02.evaluation import compare_models
from pt02.generation import compare_generation
from pt02.model import ModelConfig, TinyTransformer
from pt02.training import evaluate_loss, plot_curve, sample_text, sample_windows, write_json


def main(argv=None, *, home=HERE, default_bpe=None, include_evaluation=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="200 шагов вместо 2000")
    parser.add_argument("--bpe", type=Path, default=default_bpe or HERE.parent / "pt01/artifacts/bpe.json")
    parser.add_argument("--artifacts", type=Path, default=home / "artifacts")
    parser.add_argument("--checkpoint", type=Path, default=home / "model.pt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="загрузить обученные веса и сравнить генерацию")
    mode.add_argument("--evaluate-only", action="store_true", help="загрузить обученные веса и сравнить с биграммой")
    parser.add_argument("--steps", type=int, help="число шагов для отдельной проверки запуска")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.evaluate_only and not include_evaluation:
        parser.error("Сравнение с биграммой запускается через pt02/run.py")
    steps = args.steps if args.steps is not None else (200 if args.fast else 2000)
    if steps < 1 or args.threads < 1:
        parser.error("Число шагов и потоков должно быть положительным")
    batch = 16 if args.fast else 32
    interval = 25 if args.fast else 100
    sample_steps = {0, 50, 100, steps} if args.fast else {0, 100, 500, steps}
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    def save_generation(model, tokenizer):
        result = compare_generation(model, tokenizer, 40 if args.fast else 80, args.seed)
        result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        result["bpe_sha256"] = hashlib.sha256(args.bpe.read_bytes()).hexdigest()
        write_json(artifacts / "generation.json", result)
        for row in result["runs"]:
            print(f"{row['name']}: {row['text']}", flush=True)
            if row["name"] == "greedy":
                print(f"Первый повтор четверки: {row['loop_start']} (позиция от нуля среди новых токенов)", flush=True)

    def save_evaluation(model, tokenizer, train_words, val_words):
        report, examples = compare_models(model, tokenizer, train_words, val_words, 300 if args.fast else 2000)
        report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        report["bpe_sha256"] = hashlib.sha256(args.bpe.read_bytes()).hexdigest()
        write_json(artifacts / "eval.json", report)
        write_json(artifacts / "eval_word_examples.json", examples)
        for name in ("bigram", "tiny"):
            result = report[name]
            print(f"{name}: {result['tokens']} предсказаний, "
                  f"NLL={result['nats_total']:.4f}, нат/слово={result['nats_per_word']:.4f}, "
                  f"перплексия на слово={result['ppl_per_word']:.2f}", flush=True)

    if args.generate_only or args.evaluate_only:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["training"]["training_status"] != "completed":
            raise ValueError("Обучение загруженных весов не завершено")
        if hashlib.sha256(args.bpe.read_bytes()).hexdigest() != saved["training"]["bpe_sha256"]:
            raise ValueError("Правила BPE отличаются от использованных при обучении")
        tokenizer = BPETokens(args.bpe)
        if tokenizer.vocabulary != saved["vocabulary"]:
            raise ValueError("Номера токенов отличаются от использованных при обучении")
        model = TinyTransformer(ModelConfig(**saved["config"]))
        model.load_state_dict(saved["state_dict"])
        if args.generate_only:
            save_generation(model, tokenizer)
        else:
            raw_text = load_corpus()
            if hashlib.sha256(raw_text.encode()).hexdigest() != saved["training"]["corpus_sha256"]:
                raise ValueError("Книга отличается от использованной при обучении")
            train_words, val_words = split_words(tokenize_words(raw_text))
            save_evaluation(model, tokenizer, train_words, val_words)
        return

    print("Подготовка книги и BPE", flush=True)
    raw_text = load_corpus()
    words = tokenize_words(raw_text)
    train_words, val_words = split_words(words)
    tokenizer = BPETokens(args.bpe)
    train_tokens = torch.tensor(tokenizer.encode_words(train_words), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode_words(val_words), dtype=torch.long)
    config = ModelConfig(vocab_size=len(tokenizer.vocabulary))
    model = TinyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Отдельные генераторы не дают проверкам и образцам менять обучающие окна.
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    val_x, val_y, val_starts = sample_windows(
        val_tokens, 20, config.context, torch.Generator().manual_seed(args.seed + 2)
    )
    initial_x, initial_y, initial_starts = sample_windows(
        train_tokens, 20, config.context, torch.Generator().manual_seed(args.seed + 3)
    )
    metadata = {
        **asdict(config), "n_params": model.count_parameters(), "steps": steps,
        "batch": batch, "lr": 1e-3, "seed": args.seed,
        "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
        "bpe_merges": len(tokenizer.bpe.merges), "train_minutes": 0.0,
        "device": "cpu", "threads": args.threads, "torch_version": str(torch.__version__),
        "train_words": len(train_words), "val_words": len(val_words),
        "completed_steps": 0, "training_status": "running", "weight_decay": 0.01,
        "corpus_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        "bpe_sha256": hashlib.sha256(args.bpe.read_bytes()).hexdigest(),
        "val_window_starts": val_starts.tolist(),
        "initial_train_window_starts": initial_starts.tolist(),
        "train_loss_policy": "step 0: 20 fixed train windows; later: mean batch loss since previous measurement",
        "val_loss_policy": "mean token loss on the same 20 validation windows; no weight updates",
        "timing_policy": "training loop including loss evaluation and text samples; excludes data preparation and plotting",
    }
    write_json(artifacts / "tiny_config.json", metadata)
    print(f"Слова: обучение {len(train_words)}, проверка {len(val_words)}", flush=True)
    print(f"BPE-токены: обучение {len(train_tokens)}, проверка {len(val_tokens)}", flush=True)
    print(f"Параметров {model.count_parameters():,}; шагов {steps}; батч {batch}; контекст {config.context}", flush=True)

    samples = {"prompt": "князь андрей", "temperature": 0.8, "samples": []}
    rows = []
    started = time.perf_counter()
    with (artifacts / "train_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "train_loss", "val_loss"], lineterminator="\n")
        writer.writeheader()

        def measure(step, train_loss):
            val_loss = evaluate_loss(model, val_x, val_y)
            row = {"step": step, "train_loss": train_loss, "val_loss": val_loss}
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            elapsed = time.perf_counter() - started
            metadata.update(completed_steps=step, train_minutes=elapsed / 60)
            write_json(artifacts / "tiny_config.json", metadata)
            print(f"Шаг {step:4d}: train={train_loss:.4f}, val={val_loss:.4f}, время={elapsed:.1f} с", flush=True)

        def save_sample(step):
            text = sample_text(model, tokenizer, samples["prompt"], seed=args.seed + 4)
            samples["samples"].append({"step": step, "text": text})
            write_json(artifacts / "samples.json", samples)
            print(f"Образец {step}: {text}", flush=True)

        measure(0, evaluate_loss(model, initial_x, initial_y))
        save_sample(0)
        model.train()
        interval_loss = 0.0
        interval_steps = 0
        for step in range(1, steps + 1):
            x, y, _ = sample_windows(train_tokens, batch, config.context, train_rng)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Неконечная ошибка на шаге {step}")
            loss.backward()
            optimizer.step()
            interval_loss += loss.item()
            interval_steps += 1
            if step % interval == 0 or step == steps:
                measure(step, interval_loss / interval_steps)
                interval_loss, interval_steps = 0.0, 0
            if step in sample_steps:
                save_sample(step)

    metadata.update(train_minutes=(time.perf_counter() - started) / 60,
                    training_status="completed", completed_steps=steps)
    # Локальные веса нужны для следующих экспериментов; *.pt исключены из Git.
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_name(args.checkpoint.stem + ".tmp.pt")
    torch.save({"config": asdict(config), "state_dict": model.state_dict(),
                "vocabulary": tokenizer.vocabulary, "training": metadata}, temporary)
    temporary.replace(args.checkpoint)
    write_json(artifacts / "tiny_config.json", metadata)
    plot_curve(rows, artifacts / "train_curve.png")
    save_generation(model, tokenizer)
    if include_evaluation:
        save_evaluation(model, tokenizer, train_words, val_words)
    print(f"Обучение завершено за {metadata['train_minutes']:.2f} мин; "
          f"val {rows[0]['val_loss']:.4f} -> {rows[-1]['val_loss']:.4f}; "
          f"exp(val)={math.exp(rows[-1]['val_loss']):.2f}", flush=True)
    return artifacts


if __name__ == "__main__":
    main()
