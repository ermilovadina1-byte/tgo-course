"""Проверка происхождения результатов без загрузки модели."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "lab03/artifacts"


def test_provenance_and_bpe():
    provenance = json.loads((ART / "provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        assert hashlib.sha256((ART / name).read_bytes()).hexdigest() == expected
    config = json.loads((ART / "tiny_config.json").read_text())
    assert config["training_status"] == "completed"
    assert config["completed_steps"] == config["steps"]
    assert hashlib.sha256((ROOT / "lab02/artifacts/bpe.json").read_bytes()).hexdigest() == config["bpe_sha256"]
    if provenance["origin"] == "reused_completed_pt02_run":
        assert config["steps"] == 2000
        for name in provenance["files_sha256"]:
            assert (ART / name).read_bytes() == (ROOT / provenance["source_directory"] / name).read_bytes()
    else:
        assert provenance["origin"] == "lab03_run"


def test_lab03_uses_separate_paths():
    from unittest.mock import patch
    from lab03.run import main
    with patch("pt02.run.main", return_value=None) as train:
        main(["--fast"])
    train.assert_called_once_with(["--fast"], home=ROOT / "lab03",
                                  default_bpe=ROOT / "lab02/artifacts/bpe.json",
                                  include_evaluation=False)
