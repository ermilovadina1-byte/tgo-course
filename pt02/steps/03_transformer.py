"""Пропускаем фразу через необученный трансформер и показываем формы данных."""

import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from pt02.model import ModelConfig, TinyTransformer


def main():
    previous = runpy.run_path(str(Path(__file__).with_name("01_bpe_inputs.py")))
    tokenizer, vocabulary = previous["load_bpe"](ROOT / "pt01/artifacts/bpe.json")
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    tokens = [token for word in ["князь", "андрей", "сказал"]
              for token in tokenizer.encode_word(word)]
    ids = [token_to_id[token] for token in tokens]
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)

    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = TinyTransformer(ModelConfig(vocab_size=len(vocabulary))).eval()
    print(f"PyTorch: {torch.__version__}; устройство: cpu; веса случайные, обучения нет")
    print(f"Параметров: {model.count_parameters():,}")
    print(f"Вход: {x.tolist()}, форма {tuple(x.shape)}; цели: {y.tolist()}")

    def show_shape(name):
        def hook(module, inputs, output):
            print(f"{name}: {tuple(output.shape)}")
        return hook

    handles = [model.token_embedding.register_forward_hook(show_shape("Эмбеддинги"))]
    handles += [block.register_forward_hook(show_shape(f"После блока {i + 1}"))
                for i, block in enumerate(model.blocks)]
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocabulary)), y.reshape(-1))
        probabilities = logits.softmax(dim=-1)
    for handle in handles:
        handle.remove()
    print(f"Логиты: {tuple(logits.shape)}")
    print(f"Ошибка: {loss.item():.6f}; равномерный выбор: {math.log(len(vocabulary)):.6f}")
    for pos, target in enumerate(y[0].tolist()):
        print(f"Позиция {pos}: правильный токен {vocabulary[target]}, "
              f"вероятность {probabilities[0, pos, target].item():.6f}")

    with torch.no_grad():
        embedded = model.token_embedding(x) + model.position_embedding(torch.arange(x.size(1)))
        block = model.blocks[0]
        _, weights = block.attention(block.attention_norm(embedded), return_weights=True)
    print("Внимание первой головы первого блока (строка смотрит на столбцы):")
    for row in weights[0, 0].tolist():
        print([round(value, 6) for value in row])


if __name__ == "__main__":
    main()
