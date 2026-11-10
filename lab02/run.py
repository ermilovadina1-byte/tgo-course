"""Lab02: BPE, token budgets, paragraph retrieval and attention.

Full model inference belongs on Colab / a remote runner, never the local laptop.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pt01.run import load_corpus, tokenize_words, evaluate_bpe

QWEN = 'Qwen/Qwen3-0.6B-Base'
E5 = 'intfloat/multilingual-e5-small'
REVISIONS = {QWEN: 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
             E5: '614241f622f53c4eeff9890bdc4f31cfecc418b3'}
SENTENCE = 'Врач осмотрел пациента и сказал, что его анализы готовы.'
INSTRUCTION = 'Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ.'
ENGLISH = ('The prince had been talking for almost an hour, and the guests were growing tired. '
           'Anna looked at the clock, then at the window, then at the clock again. '
           'Nobody dared to leave before the host, and the host did not seem to notice the time at all. '
           'A servant brought tea, and the conversation turned, as it always did in that house, '
           'to the war and to the emperor.')


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@lru_cache(maxsize=1)
def qwen_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN, revision=REVISIONS[QWEN])


def count_tokens(text: str) -> int:
    return len(qwen_tokenizer()(text, add_special_tokens=False)['input_ids'])


def select_paragraphs(text: str, limit: int) -> list[dict]:
    selected = [p.strip() for p in text.splitlines() if 40 <= len(tokenize_words(p)) <= 120]
    return [{'id': i, 'text': p} for i, p in enumerate(selected[:limit])]


def span_positions(offsets, start, end):
    return [i for i, (left, right) in enumerate(offsets) if left < end and right > start]


def masked_mean(hidden, mask):
    # Accumulate in float32 even if inference uses float16.
    weights = mask.unsqueeze(-1).float()
    return (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1)


def token_budget(words):
    tok = qwen_tokenizer()
    result = {'tokenizer': QWEN, 'vocab_size': len(tok), 'context_tokens': 4096}
    for lang, text, n_words in [('ru', ' '.join(words[500:600]), 100),
                                ('en', ENGLISH, len(ENGLISH.split()))]:
        ids = tok(text, add_special_tokens=False)['input_ids']
        ratio = len(ids) / n_words
        result[lang] = dict(words=n_words, tokens=len(ids), tokens_per_word=ratio,
                            words_in_context=int(4096 / ratio), text=text)
        if lang == 'ru':
            result['example_ru'] = tok.convert_ids_to_tokens(ids)
    return result


class ParagraphSearch:
    """Reusable search(query, k); answer phrases never enter model inputs."""

    def __init__(self, model, tokenizer, paragraphs, method, batch_size=4):
        self.model, self.tokenizer = model, tokenizer
        self.paragraphs, self.method = paragraphs, method
        self.batch_size = batch_size
        self.vectors = None
        self.truncated = 0

    def encode(self, texts, query=False):
        import torch
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            part = texts[start:start + self.batch_size]
            if self.method == 'e5':
                part = [('query: ' if query else 'passage: ') + x for x in part]
            lengths = [len(x) for x in self.tokenizer(part)['input_ids']]
            self.truncated += sum(n > 512 for n in lengths) if self.method == 'e5' else 0
            kwargs = dict(padding=True, return_tensors='pt')
            if self.method == 'e5':
                kwargs.update(truncation=True, max_length=512)
            else:
                kwargs['add_special_tokens'] = False
            batch = self.tokenizer(part, **kwargs).to(self.model.device)
            with torch.inference_mode():
                if self.method == 'static':
                    hidden = self.model.get_input_embeddings()(batch['input_ids'])
                elif self.method == 'contextual':
                    out = self.model(**batch, output_hidden_states=True, use_cache=False)
                    hidden = out.hidden_states[-1]
                else:
                    out = self.model(**batch)
                    hidden = out.last_hidden_state
                pooled = masked_mean(hidden, batch['attention_mask'])
                vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
            if len(texts) > 50 and (start % 40 == 0 or start + self.batch_size >= len(texts)):
                print(f'{self.method}: {min(start+self.batch_size, len(texts))}/{len(texts)}', flush=True)
        return torch.cat(vectors)

    def index(self):
        self.vectors = self.encode([p['text'] for p in self.paragraphs])
        return self

    def search(self, query: str, k: int = 5):
        if self.vectors is None:
            raise ValueError('Call index() first')
        return self.search_many([query], k)[0]

    def search_many(self, queries, k=5):
        if k < 1 or k > len(self.paragraphs):
            raise ValueError('k outside paragraph count')
        scores = self.encode(queries, query=True) @ self.vectors.T
        # Stable sorting makes exact ties deterministic.
        return scores.argsort(dim=-1, descending=True, stable=True)[:, :k].tolist()


def attention_experiments(model, tok, corpus, lengths):
    import torch
    encoded = tok(SENTENCE, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')
    positions = {}
    for name, word in [('pronoun_pos', 'его'), ('antecedent_pos', 'пациента')]:
        start = SENTENCE.index(word)
        positions[name] = span_positions(offsets, start, start + len(word))
    ids = torch.tensor([encoded['input_ids']], device=model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)
    rows = torch.stack([a[0, :, positions['pronoun_pos'][-1], :].float().cpu()
                        for a in out.attentions])
    shares = rows[:, :, positions['antecedent_pos']].sum(-1)
    best = int(shares.argmax())
    layer, head = divmod(best, shares.shape[1])
    weights = rows[layer, head].tolist()
    att = dict(sentence=SENTENCE, tokens=[tok.decode([i]) for i in encoded['input_ids']],
               **positions, best_layer=layer, best_head=head, weights=weights,
               share_on_antecedent=sum(weights[i] for i in positions['antecedent_pos']))
    del out, rows, shares
    # Tokenize concatenation once: prefix boundaries come from character offsets,
    # not from assuming tokenize(A+B) == tokenize(A)+tokenize(B).
    combined = tok(INSTRUCTION + '\n' + corpus[:40000], add_special_tokens=False,
                   return_offsets_mapping=True)
    instruction_pos = span_positions(combined['offset_mapping'], 0, len(INSTRUCTION))
    instruction_pos = [i for i in instruction_pos if i != 0]
    dilution, evidence = [], []
    for length in lengths:
        token_ids = combined['input_ids'][:length]
        assert len(token_ids) == length and max(instruction_pos) < length
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([token_ids], device=model.device),
                        output_attentions=True, use_cache=False)
        # Keep only final rows on CPU; release large full attention tensors each run.
        row = torch.stack([a[0, :, -1, :].float().mean(0).cpu() for a in out.attentions]).mean(0)
        dilution.append(dict(context_len=length, instruction_len=len(instruction_pos),
                             share_on_instruction=float(row[instruction_pos].sum()),
                             sink_share=float(row[0]), uniform_share=len(instruction_pos)/length))
        evidence.append(dict(context_len=length, token_ids=token_ids,
                             instruction_positions=instruction_pos, mean_weights=row.tolist()))
        print('Attention:', dilution[-1], flush=True)
        del out
    return att, dilution, evidence


def make_plots(art, attention, dilution, search):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    colors = ['#d66b27' if i in attention['antecedent_pos'] else '#3574a8'
              for i in range(len(attention['tokens']))]
    ax.bar(range(len(colors)), attention['weights'], color=colors)
    labels = [f'{i}: {s.replace(" ", "·")}' for i, s in enumerate(attention['tokens'])]
    ax.set_xticks(range(len(colors)), labels, rotation=60, ha='right')
    ax.set(ylabel='Вес внимания', xlabel='Токен предложения',
           title=f'«его» смотрит на «пациента»: слой {attention["best_layer"]}, голова {attention["best_head"]} (с нуля)')
    fig.savefig(art/'attention_map.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    lengths = [r['context_len'] for r in dilution]
    for key, label, style in [('share_on_instruction', 'Инструкция без первого токена', 'o-'),
                              ('uniform_share', 'Равномерное внимание', 's--'),
                              ('sink_share', 'Первый токен (sink)', '^:')]:
        axes[0].plot(lengths, [r[key] for r in dilution], style, label=label)
    axes[0].set(xlabel='Токены во всём входе', ylabel='Доля внимания', title='Внимание последнего токена')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Доля внимания (логарифмическая шкала)')
    axes[0].legend(fontsize=9)
    values = search['recall_at_k']
    axes[1].bar(values.keys(), values.values(), color=['#8b959e', '#3574a8', '#27836a'])
    for i, v in enumerate(values.values()):
        axes[1].text(i, v + .02, f'{round(v*12)}/12', ha='center')
    axes[1].set(ylim=(0, 1.15), ylabel='Recall@5', title=f'Поиск среди {search["n_paragraphs"]} абзацев')
    fig.savefig(art/'search_and_dilution.png', dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--artifacts', type=Path, default=HERE/'artifacts')
    args = ap.parse_args(argv)
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently fall back to local CPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    art = args.artifacts.resolve(); art.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Preserve the source text exactly on both fresh and cached runs.
    # read_text() would silently convert CRLF to LF and change model tokens.
    load_corpus()
    corpus = (ROOT/'pt01/corpus.txt').read_bytes().decode('utf-8-sig')
    words = tokenize_words(corpus)
    bpe = evaluate_bpe(Counter(words), 200 if args.fast else 1500)
    save_json(art/'bpe.json', bpe)
    budget = token_budget(words); save_json(art/'tokens_per_word.json', budget)
    print('Token budgets:', {lang: budget[lang]['tokens_per_word'] for lang in ('ru','en')}, flush=True)
    paragraphs = select_paragraphs(corpus, 60 if args.fast else 500)
    save_json(art/'paragraphs.json', paragraphs)
    queries = json.loads((HERE/'queries.json').read_text())
    for q in queries:
        q['answer_id'] = next(p['id'] for p in paragraphs if q['answer_phrase'] in p['text'])
        q['top'] = {}
    tok = qwen_tokenizer()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(QWEN, revision=REVISIONS[QWEN], torch_dtype=dtype, attn_implementation='eager').to(args.device).eval()
    revisions = {QWEN: model.config._commit_hash}
    vector_evidence = {}
    for method in ('static', 'contextual'):
        engine = ParagraphSearch(model, tok, paragraphs, method, batch_size=4 if args.device=='cuda' else 1).index()
        tops = engine.search_many([q['query'] for q in queries])
        for q, top in zip(queries, tops): q['top'][method] = top
        vector_evidence[method] = engine.vectors.numpy()
        vector_evidence[method + '_queries'] = engine.encode([q['query'] for q in queries], query=True).numpy()
        del engine
    attention, dilution, evidence = attention_experiments(model, tok, corpus, [64,256,768] if args.fast else [128,512,1024])
    del model; gc.collect()
    if args.device == 'cuda': torch.cuda.empty_cache()
    tok_e5 = AutoTokenizer.from_pretrained(E5, revision=REVISIONS[E5])
    model_e5 = AutoModel.from_pretrained(E5, revision=REVISIONS[E5], torch_dtype=dtype).to(args.device).eval()
    revisions[E5] = model_e5.config._commit_hash
    engine = ParagraphSearch(model_e5, tok_e5, paragraphs, 'e5', batch_size=8 if args.device=='cuda' else 2).index()
    tops = engine.search_many([q['query'] for q in queries])
    for q, top in zip(queries, tops): q['top']['e5'] = top
    vector_evidence['e5'] = engine.vectors.numpy()
    vector_evidence['e5_queries'] = engine.encode([q['query'] for q in queries], query=True).numpy()
    search = dict(k=5, n_paragraphs=len(paragraphs), queries=queries,
                  recall_at_k={m: sum(q['answer_id'] in q['top'][m] for q in queries)/len(queries)
                               for m in ('static','contextual','e5')})
    import csv
    import numpy as np
    np.savez_compressed(art/'retrieval_vectors.npz', **vector_evidence)
    save_json(art/'search.json', search)
    save_json(art/'attention_map.json', attention)
    save_json(art/'dilution_evidence.json', evidence)
    with (art/'dilution.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(dilution[0])); writer.writeheader(); writer.writerows(dilution)
    make_plots(art, attention, dilution, search)
    source_files = ['lab02/run.py', 'lab02/queries.json', 'pt01/run.py']
    metadata = dict(mode='fast' if args.fast else 'full', device=args.device, dtype=str(dtype),
                    gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                    torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                    elapsed_seconds=time.perf_counter()-started, model_revisions=revisions,
                    corpus_sha256=hashlib.sha256(corpus.encode()).hexdigest(), e5_truncated_inputs=engine.truncated,
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
                    artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(art.iterdir())
                                     if p.is_file() and p.name != 'run_metadata.json'})
    save_json(art/'run_metadata.json', metadata)
    print('RESULT', json.dumps(search['recall_at_k']), flush=True)
    print('FINISHED', metadata['elapsed_seconds'], flush=True)
    return art


if __name__ == '__main__':
    main()
