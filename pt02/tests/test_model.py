"""Проверки формулы внимания, отсутствия подсматривания и обучения весов."""

import math
import unittest

import torch
from torch.nn import functional as F

from pt02.model import CausalSelfAttention, ModelConfig, TinyTransformer


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def make_model(self):
        return TinyTransformer(ModelConfig(
            vocab_size=19, d_model=16, n_layers=2, n_heads=4, d_ff=32, context=8
        )).eval()

    def test_attention_matches_hand_calculation(self):
        attention = CausalSelfAttention(ModelConfig(
            vocab_size=3, d_model=2, n_layers=1, n_heads=1, d_ff=4, context=2
        ))
        # nn.Linear хранит матрицу транспонированной относительно записи x @ W.
        with torch.no_grad():
            attention.q.weight.copy_(torch.tensor([[1., 0.], [1., -1.]]))
            attention.k.weight.copy_(torch.tensor([[2., -2.], [0., 1.]]))
            attention.v.weight.copy_(torch.tensor([[10., -10.], [0., 10.]]))
            attention.out.weight.copy_(torch.eye(2))
        result, weights = attention(torch.tensor([[[1., 0.], [1., 1.]]]), True)
        prince_share = math.exp(math.sqrt(2)) / (math.exp(math.sqrt(2)) + 1)
        expected_weights = torch.tensor([[[[1., 0.], [prince_share, 1 - prince_share]]]])
        expected_values = torch.tensor([[[10., 0.], [10 * prince_share, 10 * (1 - prince_share)]]])
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(result, expected_values)

    def test_future_tokens_cannot_change_earlier_predictions(self):
        model = self.make_model()
        original = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = torch.tensor([[1, 2, 3, 9, 8, 7]])
        with torch.no_grad():
            before = model(original)
            after = model(changed)
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        self.assertGreater((before[:, 3:] - after[:, 3:]).abs().max().item(), 1e-5)

    def test_each_head_has_normalized_causal_weights(self):
        attention = self.make_model().blocks[0].attention
        _, weights = attention(torch.randn(2, 6, 16), return_weights=True)
        self.assertEqual(tuple(weights.shape), (2, 4, 6, 6))
        self.assertEqual(torch.count_nonzero(torch.triu(weights, diagonal=1)).item(), 0)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 6))

    def test_batch_and_prefix_give_the_same_predictions(self):
        model = self.make_model()
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        with torch.no_grad():
            together = model(ids)
            for row in range(2):
                torch.testing.assert_close(together[row:row + 1], model(ids[row:row + 1]))
            for length in range(1, 5):
                torch.testing.assert_close(together[:, :length], model(ids[:, :length]))

    def test_cross_entropy_reaches_every_trainable_part(self):
        model = self.make_model().train()
        sequence = torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
        x, y = sequence[:, :-1], sequence[:, 1:]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 4, 19))
        loss = F.cross_entropy(logits.reshape(-1, 19), y.reshape(-1))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        with torch.no_grad():
            new_loss = F.cross_entropy(model(x).reshape(-1, 19), y.reshape(-1))
        self.assertLess(new_loss.item(), loss.item())


if __name__ == "__main__":
    unittest.main()
