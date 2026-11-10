"""Очная lab03: общий с pt02 трансформер, отдельные результаты и пути.

Полный запуск обучает модель 2000 шагов, --fast выполняет 200 шагов.
Запускайте обучение в облаке или GitHub Actions.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from pt02.run import main as train

    artifacts = train(argv, home=HERE, default_bpe=ROOT / "lab02/artifacts/bpe.json",
                      include_evaluation=False)
    if artifacts is not None:
        files = ["tiny_config.json", "train_curve.csv", "samples.json", "generation.json", "train_curve.png"]
        provenance = {
            "origin": "lab03_run", "implementation": "pt02",
            "files_sha256": {name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
                             for name in files},
        }
        (artifacts / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
