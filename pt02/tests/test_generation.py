"""Порог top-p, температура, жадный выбор и границы повторов."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pt02.generation import TextGenerator, first_repeat, sampling_logits


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0., 2., 1.]))
        self.config = SimpleNamespace(context=2)
        self.window_lengths = []

    def forward(self, ids):
        self.window_lengths.append(ids.size(1))
        return self.logits.expand(ids.size(0), ids.size(1), -1)


class ToyTokenizer:
    def encode(self, text):
        return [0, 2]

    def decode(self, ids):
        return " ".join(map(str, ids))


class GenerationTests(unittest.TestCase):
    def test_top_p_keeps_the_token_crossing_the_threshold(self):
        probabilities = torch.tensor([0.05, 0.3, 0.5, 0.15], dtype=torch.float64)
        filtered = sampling_logits(probabilities.log(), 1.0, 0.9)
        self.assertEqual(torch.isfinite(filtered).tolist(), [False, True, True, True])
        torch.testing.assert_close(filtered.softmax(-1), torch.tensor(
            [0., 0.3 / 0.95, 0.5 / 0.95, 0.15 / 0.95], dtype=torch.float64
        ))

    def test_exact_threshold_and_full_probability(self):
        logits = torch.zeros(4, dtype=torch.float64)
        # Две равновероятные позиции уже дают ровно 0.5, третья не нужна.
        filtered = sampling_logits(logits, 1.0, 0.5)
        self.assertEqual(torch.isfinite(filtered).tolist(), [True, True, False, False])
        torch.testing.assert_close(sampling_logits(logits, 1.0, 1.0), logits)
        self.assertEqual(torch.isfinite(sampling_logits(logits, 1.0, 0.01)).sum().item(), 1)

    def test_temperature_changes_concentration(self):
        logits = torch.tensor([0., 1., 2.])
        low = sampling_logits(logits, 0.7).softmax(-1)
        high = sampling_logits(logits, 1.5).softmax(-1)
        self.assertGreater(low[2].item(), high[2].item())
        self.assertLess(low[0].item(), high[0].item())

    def test_greedy_sampling_and_context_limit(self):
        model = ToyModel().train()
        result = TextGenerator(model, ToyTokenizer(), seed=3).generate("prompt", 8, temperature=0)
        self.assertEqual(result["ids"], [1] * 8)
        self.assertEqual(result["text"], "0 2 " + " ".join(["1"] * 8))
        self.assertEqual(result["n_tokens"], 8)
        self.assertEqual(max(model.window_lengths), 2)
        self.assertTrue(model.training)
        self.assertIsNone(model.logits.grad)
        other = TextGenerator(model, ToyTokenizer(), seed=19).generate("prompt", 8, temperature=0)
        self.assertEqual(result, other)

    def test_sampling_is_reproducible_without_changing_global_rng(self):
        model = ToyModel()
        generator = TextGenerator(model, ToyTokenizer(), seed=42)
        state = torch.get_rng_state().clone()
        a = generator.generate("prompt", 30, temperature=1.5, top_p=0.9)
        b = generator.generate("prompt", 30, temperature=1.5, top_p=0.9)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_repeat_positions_include_overlapping_groups(self):
        self.assertIsNone(first_repeat([1, 2, 3, 4]))
        self.assertEqual(first_repeat([1] * 5), 1)
        self.assertEqual(first_repeat([1, 2, 3, 4, 9, 1, 2, 3, 4]), 5)


if __name__ == "__main__":
    unittest.main()
