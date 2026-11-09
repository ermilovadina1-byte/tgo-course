"""Одинаковые цели, единицы нормировки и границы скользящего контекста."""

import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from pt02.evaluation import compare_models, token_nlls
from pt02.model import ModelConfig, TinyTransformer


class UniformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(5))
        self.config = SimpleNamespace(context=4)

    def forward(self, ids):
        return self.logits.expand(ids.size(0), ids.size(1), -1)


class Tokenizer:
    vocabulary = ["s", "eed", "aa", "un", "known"]

    def encode_word(self, word):
        return {"seed": [0, 1], "aa": [2], "unknown": [3, 4]}[word]


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uniform_model_scores_each_target_once(self):
        ids = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]
        for first_target in (1, 3, 7):
            scores = token_nlls(UniformModel(), ids, first_target, batch_size=2)
            self.assertEqual(scores[:first_target], [None] * first_target)
            self.assertEqual(len(scores), len(ids))
            self.assertAlmostEqual(math.fsum(scores[first_target:]),
                                   (len(ids) - first_target) * math.log(5), places=5)

    def test_batched_scoring_matches_individual_sliding_windows(self):
        torch.manual_seed(42)
        model = TinyTransformer(ModelConfig(
            vocab_size=9, d_model=8, n_layers=2, n_heads=2, d_ff=16, context=4
        )).train()
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 4, 6]
        actual = token_nlls(model, ids, first_target=2, batch_size=3)
        self.assertTrue(model.training)
        with torch.no_grad():
            for target in range(2, len(ids)):
                prefix = torch.tensor([ids[max(0, target - 4):target]])
                logits = model(prefix)[0, -1]
                expected = F.cross_entropy(logits.unsqueeze(0), torch.tensor([ids[target]])).item()
                self.assertAlmostEqual(actual[target], expected, places=5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_word_aggregation_excludes_entire_first_word(self):
        report, examples = compare_models(
            UniformModel(), Tokenizer(), ["seed", "aa", "seed", "aa"],
            ["seed", "aa", "unknown"], word_limit=3
        )
        self.assertEqual(report["words"], 3)
        self.assertEqual(report["scored_words"], 2)
        self.assertEqual(report["context_bpe_tokens"], 2)
        self.assertEqual(report["tiny"]["tokens"], 3)
        self.assertAlmostEqual(report["tiny"]["nats_total"], 3 * math.log(5), places=5)
        self.assertAlmostEqual(report["tiny"]["nats_per_word"], math.log(5), places=5)
        self.assertEqual(report["bigram"]["tokens"], 2)
        expected_bigram = -math.log(2.01 / 2.03) - math.log(0.01 / 1.03)
        self.assertAlmostEqual(report["bigram"]["nats_total"], expected_bigram)
        self.assertEqual(report["bigram"]["unknown_target_words"], 1)
        self.assertEqual(report["known_target_words_only"]["scored_words"], 1)
        self.assertAlmostEqual(examples["examples"][1]["tiny_nats"], 2 * math.log(5), places=5)


if __name__ == "__main__":
    unittest.main()
