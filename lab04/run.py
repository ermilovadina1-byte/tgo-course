"""Три режима Qwen. Colab: --device cuda --dtype float16. Импорт не загружает модель."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
MODEL = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
TEST_SHA = '874107c454a1f0c9f0b9313359304291de9325d16478f83883f19bf6486d45f9'
FEW_SHA = '82d9e61857861d1142f10ab6aa8188aa421f3fda1e8c6f3b561b9543611624cb'
MODES = ['zero_shot', 'few_shot', 'adapter']


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def save_predictions(path, rows):
    columns = ['idx', 'label'] + MODES + [f'raw_{m}' for m in MODES]
    columns += [f'{p}_{m}' for m in MODES for p in ('tokens', 'seconds')]
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--data')
    ap.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto')
    ap.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'])
    ap.add_argument('--artifacts', type=Path, default=HERE / 'artifacts')
    ap.add_argument('--adapter', type=Path, default=HERE / 'artifacts/adapter')
    args = ap.parse_args(argv)
    os.environ.setdefault('HF_HOME', str(HERE / '.cache/huggingface'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from lab04.data import DATASET, REVISION, load_reviews, select_reviews, selection_digest, star_counts
    from lab04.evaluation import score_ratings
    from lab04.prompting import zero_shot_messages, few_shot_messages, parse_rating

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна: выберите облачный GPU Colab')
    dtype_name = args.dtype or ('float16' if device == 'cuda' else 'float32')
    if device == 'cuda' and dtype_name == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('GPU не поддерживает BF16 аппаратно')
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.artifacts.resolve()
    out.mkdir(parents=True, exist_ok=True)
    adapter_path = args.adapter.resolve()
    weight_path = adapter_path / 'adapter_model.safetensors'
    adapter_present = weight_path.is_file()
    adapter_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest() if adapter_present else None
    train_config_path = adapter_path.parent / 'train_config.json'
    if adapter_present and train_config_path.exists():
        training = json.loads(train_config_path.read_text())
        assert training['model_revision'] == MODEL_REVISION
        assert training['adapter_sha256'] == adapter_sha
    data = load_reviews(args.data)
    selected = select_reviews(data, seed=42, fast=args.fast)
    test = data['test'].select(selected.test_indices)
    few = data['validation'].select(selected.few_shot_indices)
    examples = list(few)
    test_sha = selection_digest(test)
    assert selected.few_shot_indices == [6, 8, 3, 0, 1]
    assert selection_digest(few) == FEW_SHA
    if not args.fast:
        assert test_sha == TEST_SHA, 'Изменилась проверочная выборка'
    assert not set(test['text']) & set(few['text'])
    assert not set(test['text']) & set(data['train'].select(selected.train_indices)['text'])
    labels = [int(y) + 1 for y in test['label']]
    records = [{'idx': idx, 'label': y} for idx, y in enumerate(labels)]
    print(f'Отзывов: {len(test)}, классы: {star_counts(test["label"])}', flush=True)
    print(f'SHA256 теста: {test_sha}; примеры validation: {selected.few_shot_indices}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=dtype).to(device).eval()
    base.config.use_cache = True
    frozen = [(p, p._version) for p in base.parameters()]
    print(f'Загружена {MODEL}, устройство {device}, dtype {dtype}', flush=True)

    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

    def predict(model, messages):
        synchronize()
        started = time.perf_counter()
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=3,
                                    temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
        ids = output[0, inputs['input_ids'].shape[1]:].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        pred = parse_rating(raw)
        synchronize()
        return {'prediction': pred, 'raw_answer': raw, 'output_ids': ids,
                'prompt_tokens': inputs['input_ids'].shape[1],
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'seconds': time.perf_counter() - started}

    metrics = {
        'model': MODEL, 'model_revision': MODEL_REVISION, 'dataset': DATASET,
        'dataset_revision': REVISION if not (args.data or os.environ.get('LAB_DATA')) else None,
        'n_test': len(test), 'fast': args.fast, 'test_sha256': test_sha,
        'class_counts': star_counts(test['label']), 'few_shot_indices': selected.few_shot_indices,
        'few_shot_examples_sha256': selection_digest(few),
        'adapter_present': adapter_present, 'adapter_sha256': adapter_sha,
        'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'dtype': str(dtype), 'batch_size': 1, 'do_sample': False, 'max_new_tokens': 3,
        'enable_thinking': False, 'truncation': False, 'status': 'running', 'modes': {},
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'datasets')},
        'policy': {
            'seconds': 'Mean synchronized wall time including chat template, tokenization, generation and parsing; excludes loading, warmup, file writes',
            'memory': 'CUDA peak allocated bytes / 1e9, includes weights; CPU/MPS peak not measured',
            'invalid': 'Wrong for accuracy and F1; excluded from MAE and counted separately',
            'test': 'First 1000 test rows, or first 24 in fast mode; no prompt tuning',
            'cache': 'KV cache only within one request; no shared cache between reviews'},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), HERE / 'data.py', HERE / 'prompting.py', HERE / 'evaluation.py')}}
    save_json(out / 'metrics.json', metrics)
    modes = MODES if adapter_present else MODES[:2]
    if not adapter_present:
        print('Адаптер отсутствует: будут оценены только два базовых режима', flush=True)
    model = base
    with (out / 'requests.jsonl').open('w', encoding='utf-8') as journal:
        for mode in modes:
            if mode == 'adapter':
                model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).eval()
                assert not any(p.requires_grad for p in model.parameters())
            builder = (lambda text: few_shot_messages(text, examples)) if mode == 'few_shot' else zero_shot_messages
            predict(model, builder('Заказ доставили вовремя, всё понравилось.'))
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            mode_started = time.perf_counter()
            for idx, row in enumerate(test):
                result = predict(model, builder(row['text']))
                records[idx].update({mode: result['prediction'], f'raw_{mode}': result['raw_answer'],
                                     f'tokens_{mode}': result['prompt_tokens'], f'seconds_{mode}': result['seconds']})
                journal.write(json.dumps({'idx': idx, 'label': labels[idx], 'mode': mode, **result}, ensure_ascii=False) + '\n')
                if (idx + 1) % 100 == 0 or idx + 1 == len(test):
                    journal.flush()
                    correct = sum(r[mode] == r['label'] for r in records[:idx + 1])
                    save_predictions(out / 'predictions.csv', records)
                    save_json(out / 'progress.json', {'mode': mode, 'completed': idx + 1, 'total': len(test)})
                    print(f'{mode}: {idx + 1}/{len(test)}, совпало {correct}, {time.perf_counter() - mode_started:.1f} с', flush=True)
            score = score_ratings(labels, [r[mode] for r in records])
            score['sec_per_example'] = sum(r[f'seconds_{mode}'] for r in records) / len(test)
            score['prompt_tokens'] = sum(r[f'tokens_{mode}'] for r in records) / len(test)
            score['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else None
            metrics['modes'][mode] = score
            save_json(out / 'metrics.json', metrics)
            print(f"ИТОГ {mode}: accuracy={score['accuracy']:.4f}, macro-F1={score['f1_macro']:.4f}, MAE={score['mae_stars']}, invalid={score['invalid']}", flush=True)
    assert all(p._version == v for p, v in frozen)
    assert all(p.grad is None for p in model.parameters())
    if adapter_present:
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest() == adapter_sha
    metrics['weights_unchanged'] = True
    metrics['status'] = 'completed'
    metrics['paired_comparison'] = {
        mode: {'fixed_vs_zero_shot': sum(r['zero_shot'] != r['label'] == r[mode] for r in records),
               'broken_vs_zero_shot': sum(r['zero_shot'] == r['label'] != r[mode] for r in records)}
        for mode in modes if mode != 'zero_shot'}
    save_json(out / 'metrics.json', metrics)
    save_predictions(out / 'predictions.csv', records)
    print('Все режимы завершены, веса не менялись:', out, flush=True)


if __name__ == '__main__':
    main()
