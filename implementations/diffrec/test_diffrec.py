"""Regression checks for the reproduction's weighting, loss history and ranking."""

from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
import torch

from data import Interactions, add_random_noise, temporal_weights
from diffusion_loss import training_loss, update_history
from prepare import UPSTREAM
from ranking import evaluate


class DiffRecTests(unittest.TestCase):
    def test_temporal_weights_follow_each_users_file_order(self):
        users = np.array([0, 2, 0, 1, 2, 0])
        np.testing.assert_allclose(temporal_weights(users, 0.1), [0.1, 0.1, 0.55, 0.1, 1.0, 1.0])

    def test_random_noise_keeps_test_and_preserves_split_boundaries(self):
        train = csr_matrix([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
        valid = csr_matrix([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]])
        test = csr_matrix([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
        original = Interactions(train, valid, test, train)
        noisy = add_random_noise(original, ratio=0.5, seed=1)
        repeated = add_random_noise(original, ratio=0.5, seed=1)
        self.assertEqual((noisy.train - repeated.train).nnz, 0)
        self.assertEqual((noisy.valid - repeated.valid).nnz, 0)
        self.assertEqual((noisy.test - test).nnz, 0)
        self.assertEqual(noisy.train.nnz, 6)
        self.assertEqual(noisy.valid.nnz, 3)
        self.assertEqual(noisy.train.multiply(noisy.valid).nnz, 0)
        self.assertEqual((noisy.train + noisy.valid).multiply(test).nnz, 0)

    def test_timestep_history_keeps_latest_losses(self):
        diffusion = SimpleNamespace(Lt_history=torch.zeros((3, 3)), Lt_count=torch.zeros(3, dtype=torch.long))
        histories = [deque(maxlen=3) for _ in range(3)]
        for steps, losses in [([0, 2, 0], [1, 2, 3]), ([2, 0, 0, 0, 1, 0], [4, 5, 6, 7, 8, 9])]:
            update_history(diffusion, torch.tensor(steps), torch.tensor(losses, dtype=torch.float32))
            for step, loss in zip(steps, losses):
                histories[step].append(loss)
            for row, expected in enumerate(histories):
                count = len(expected)
                self.assertEqual(diffusion.Lt_count[row].item(), count)
                if count:
                    self.assertEqual(diffusion.Lt_history[row, -count:].tolist(), list(expected))

    def test_weighted_loss_matches_upstream(self):
        source = UPSTREAM / "DiffRec" / "models" / "gaussian_diffusion.py"
        if not source.is_file():
            self.skipTest("execute prepare.py para comparar com o código oficial")
        spec = spec_from_file_location("official_diffusion", source)
        official = module_from_spec(spec)
        spec.loader.exec_module(official)
        settings = (official.ModelMeanType.START_X, "linear-var", 0.01, 0.0005, 0.005, 5, torch.device("cpu"))
        original = official.GaussianDiffusion(*settings, history_num_per_term=2)
        local = official.GaussianDiffusion(*settings, history_num_per_term=2)
        batch = torch.eye(16)

        def predict(corrupted, steps):
            return corrupted * 0.5 + steps[:, None] * 0.02

        for seed in range(4):
            torch.manual_seed(seed)
            expected = original.training_losses(predict, batch, reweight=True)["loss"].mean()
            torch.manual_seed(seed)
            result, _ = training_loss(local, predict, batch, reweight=True)
            torch.testing.assert_close(result, expected)

    def test_ranking_masks_history_and_counts_empty_users(self):
        inputs = csr_matrix((2, 24))
        targets = csr_matrix(([1, 1], ([0, 0], [1, 12])), shape=inputs.shape)
        mask = csr_matrix(([1], ([0], [0])), shape=inputs.shape)

        def predict(batch):
            return -torch.arange(24, dtype=torch.float32).repeat(batch.shape[0], 1)

        scores = evaluate(predict, inputs, targets, mask, batch_size=1)
        ideal = 1 + 1 / np.log2(3)
        self.assertEqual(scores["recall@10"], 0.25)
        self.assertEqual(scores["recall@20"], 0.5)
        self.assertEqual(scores["ndcg@10"], round(0.5 / ideal, 4))
        self.assertEqual(scores["ndcg@20"], round((1 + 1 / np.log2(13)) / (2 * ideal), 4))


if __name__ == "__main__":
    unittest.main()
