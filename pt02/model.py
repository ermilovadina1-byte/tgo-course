"""Небольшой декодер с самостоятельно реализованным причинным вниманием."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    context: int = 128

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if self.d_model % self.n_heads:
            raise ValueError("d_model должен делиться на n_heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        # True только справа от диагонали: это будущие, запрещенные позиции.
        self.register_buffer(
            "future_mask",
            torch.ones(config.context, config.context, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        batch, length, width = x.shape
        if not 1 <= length <= self.future_mask.size(0):
            raise ValueError("Длина входа внимания вне границ контекста")

        def split_heads(projection):
            # (батч, позиции, 128) -> (батч, 4 головы, позиции, 32).
            return projection(x).reshape(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        queries = split_heads(self.q)
        keys = split_heads(self.k)
        values = split_heads(self.v)
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.future_mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        mixed = weights @ values
        # Собираем результаты четырех голов в один вектор каждой позиции.
        joined = mixed.transpose(1, 2).contiguous().reshape(batch, length, width)
        result = self.out(joined)
        if return_weights:
            return result, weights
        return result


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Нормируем вход каждой ветви, а ее результат прибавляем к потоку x.
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("Ожидаются целые номера torch.long формы (батч, позиции)")
        batch, length = ids.shape
        if batch < 1 or not 1 <= length <= self.config.context:
            raise ValueError("Нужен непустой батч с длиной окна от 1 до context")
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        # Возвращаем логиты: cross_entropy сама выполняет нужное преобразование.
        return self.lm_head(self.final_norm(x))

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
