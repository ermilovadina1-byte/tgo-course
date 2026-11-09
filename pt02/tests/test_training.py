"""Проверяем сдвиг целей и отсутствие обновления весов при оценке."""

import unittest

import torch
from torch.nn import functional as F

from pt02.model import ModelConfig, TinyTransformer
from pt02.training import evaluate_loss, sample_windows


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_windows_use_next_tokens_and_stay_in_the_given_split(self):
        # Каждый номер здесь равен позиции в исходном потоке.
        stream = torch.arange(100, 140)
        x, y, starts = sample_windows(stream, 32, 8, torch.Generator().manual_seed(42))
        self.assertEqual(tuple(x.shape), (32, 8))
        torch.testing.assert_close(y, x + 1)
        torch.testing.assert_close(x[:, 0], starts + 100)
        self.assertGreaterEqual(x.min().item(), 100)
        self.assertLessEqual(y.max().item(), 139)
        _, last_y, _ = sample_windows(torch.arange(9), 1, 8, torch.Generator())
        self.assertEqual(last_y[0, -1].item(), 8)

    def test_evaluation_does_not_train_or_consume_randomness(self):
        torch.manual_seed(42)
        model = TinyTransformer(ModelConfig(
            vocab_size=19, d_model=16, n_layers=2, n_heads=4, d_ff=32, context=8
        )).train()
        x, y, _ = sample_windows(torch.arange(60) % 19, 5, 8, torch.Generator().manual_seed(7))
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        random_state = torch.get_rng_state().clone()
        measured = evaluate_loss(model, x, y, batch_size=2)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]), name)
            self.assertIsNone(value.grad, name)
        with torch.no_grad():
            reference = F.cross_entropy(model(x).reshape(-1, 19), y.reshape(-1)).item()
        self.assertAlmostEqual(measured, reference, places=5)
        self.assertAlmostEqual(measured, evaluate_loss(model, x, y, batch_size=4), places=5)


if __name__ == "__main__":
    unittest.main()
