"""Собственные проверки lab01: преподаватель оценивает ноутбук и отчёт."""
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_full_results():
    r = json.loads((ROOT / "lab01/artifacts/bigram.json").read_text())
    reference = json.loads((ROOT / "pt01/artifacts/bigram.json").read_text())
    assert r["words"] > 100000
    assert r["train_words"] + r["val_words"] == r["words"]
    assert 0.099 < r["val_words"] / r["words"] < 0.101
    assert r["prediction_count"] == r["val_words"] - 1
    assert len(r["top20"]) == 20
    assert len(r["generation"]) == len({g["start"] for g in r["generation"]}) == 5
    assert {g["temperature"] for g in r["temperature_generation"]} == {0.5, 1.0, 2.0}
    for g in r["generation"] + r["temperature_generation"]:
        assert len(g["text"].split()) == 30
        assert g["text"].split()[0] == g["start"]
    for alpha, value in r["perplexity"].items():
        assert math.isclose(value, reference["perplexity"][str(float(alpha)).removesuffix(".0")], rel_tol=1e-12)
    lk = r["likelihood"]
    assert Counter(lk["sentence"].split()) == Counter(lk["shuffled"].split())
    for name in ["sentence", "shuffled"]:
        trace = lk[name + "_trace"]
        assert math.isclose(sum(math.log(t["prob"]) for t in trace), lk["logprob_" + name])
        assert r["unsmoothed"][name]["unseen_pairs"] == sum(t["train_pair_count"] == 0 for t in trace)
        assert not r["unsmoothed"][name]["finite"]
    assert lk["logprob_sentence"] > lk["logprob_shuffled"]


def test_executed_notebook():
    import nbformat
    n = nbformat.read(ROOT / "lab01/lab01.ipynb", as_version=4)
    nbformat.validate(n)
    cells = [c for c in n.cells if c.cell_type == "code"]
    assert [c.execution_count for c in cells] == list(range(1, len(cells) + 1))
    assert all(c.outputs for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    assert any("image/png" in o.get("data", {}) for c in cells for o in c.outputs)
