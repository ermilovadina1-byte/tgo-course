"""Lightweight checks of saved inference evidence, without loading models."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'lab02/artifacts'


def read(name):
    return json.loads((ART/name).read_text())


def test_hashes_and_full_or_fast_scope():
    meta=read('run_metadata.json')
    for name, expected in meta['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    for name, expected in meta['artifact_sha256'].items():
        assert hashlib.sha256((ART/name).read_bytes()).hexdigest()==expected, name
    assert len(read('paragraphs.json'))==({'full':500,'fast':60}[meta['mode']])
    assert read('bpe.json')['num_merges']>=({'full':1500,'fast':200}[meta['mode']])


def test_retrieval_ranks_recomputed_from_vectors():
    search=read('search.json')
    with np.load(ART/'retrieval_vectors.npz',allow_pickle=False) as vectors:
        for method in ['static','contextual','e5']:
            docs=vectors[method];queries=vectors[method+'_queries']
            assert docs.shape[0]==search['n_paragraphs']
            assert queries.shape[0]==12
            assert np.isfinite(docs).all() and np.isfinite(queries).all()
            assert np.linalg.norm(docs,axis=1)==pytest.approx(np.ones(len(docs)),abs=1e-5)
            top=np.argsort(-(queries@docs.T),axis=-1,kind='stable')[:,:5].tolist()
            assert top==[q['top'][method] for q in search['queries']]


def test_dilution_recomputed_from_saved_attention_rows():
    rows=list(csv.DictReader((ART/'dilution.csv').open()))
    evidence=read('dilution_evidence.json')
    assert len(rows)==len(evidence)==3
    for row,ev in zip(rows,evidence):
        weights=ev['mean_weights'];pos=ev['instruction_positions']
        assert 0 not in pos
        assert len(weights)==len(ev['token_ids'])==int(row['context_len'])
        assert sum(weights)==pytest.approx(1,abs=1e-3)
        assert min(weights)>=0
        assert len(pos)==int(row['instruction_len'])
        assert sum(weights[i] for i in pos)==pytest.approx(float(row['share_on_instruction']),abs=1e-6)
        assert weights[0]==pytest.approx(float(row['sink_share']),abs=1e-6)
