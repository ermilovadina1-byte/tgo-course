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


def test_generate_only_returns_output_directory(tmp_path):
    """Повторная генерация должна позволять обновить SHA256, без загрузки весов в тесте."""
    from unittest.mock import MagicMock, patch
    from pt02.run import main
    bpe, checkpoint = tmp_path / "bpe.json", tmp_path / "model.pt"
    bpe.write_text("{}")
    checkpoint.write_bytes(b"test checkpoint")
    saved = {"training": {"training_status": "completed",
                          "bpe_sha256": hashlib.sha256(bpe.read_bytes()).hexdigest()},
             "vocabulary": ["a"], "config": {}, "state_dict": {}}
    tokenizer = MagicMock(vocabulary=["a"])
    with patch("pt02.run.torch.load", return_value=saved), \
         patch("pt02.run.BPETokens", return_value=tokenizer), \
         patch("pt02.run.ModelConfig"), patch("pt02.run.TinyTransformer"), \
         patch("pt02.run.compare_generation", return_value={"runs": []}):
        output = main(["--generate-only", "--bpe", str(bpe), "--checkpoint", str(checkpoint)],
                      home=tmp_path, include_evaluation=False)
    assert output == tmp_path / "artifacts"
    assert (output / "generation.json").is_file()
