import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from task_aware_compression.compressor import compress_adaptive, decode_adaptive
    from task_aware_compression.learned_allocation import (
        FEATURE_NAMES, TileRateAllocator, budgeted_softmax, build_candidate_bank,
        compress_learned, select_under_budget,
        select_near_budget,
    )
    from task_aware_compression.types import BoundingBox, CompressionConfig, Detection


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class LearnedAllocationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(17).integers(0, 256, (57, 77, 3), dtype=np.uint8)
        self.detections = [Detection(BoundingBox(24, 8, 40, 44), .2)]
        self.config = CompressionConfig(tile_size=32, background_quality=8, person_quality=82)

    def test_zero_policy_reproduces_baseline_encoded_bytes(self):
        baseline = compress_adaptive(self.image, self.detections, self.config)
        learned = compress_learned(self.image, self.detections, TileRateAllocator(), self.config)
        self.assertEqual(learned.encoded, baseline.encoded)
        self.assertEqual(learned.file_size, baseline.file_size)

    def test_exact_costs_include_header_for_every_candidate(self):
        bank = build_candidate_bank(self.image, self.detections, self.config)
        rng = np.random.default_rng(27)
        for _ in range(6):
            levels = rng.integers(0, bank.qualities.shape[1], len(bank.rectangles))
            result = bank.compress(levels)
            expected = bank.fixed_bytes + bank.costs[np.arange(len(levels)), levels].sum()
            self.assertEqual(result.file_size, expected)
            decoded, _ = decode_adaptive(result.encoded)
            np.testing.assert_array_equal(decoded, result.image)

    def test_budget_projection_handles_nonmonotonic_costs(self):
        rng = np.random.default_rng(17)
        for _ in range(25):
            costs = rng.integers(100, 600, (12, 5))
            budget = int(costs.min(1).sum() + 400)
            scores = rng.normal(size=costs.shape)
            levels = select_under_budget(scores, costs, budget)
            self.assertLessEqual(costs[np.arange(len(costs)), levels].sum(), budget)
        with self.assertRaises(ValueError):
            select_under_budget(scores, costs, int(costs.min(1).sum()) - 1)

    def test_nonzero_policy_remains_under_real_tac_budget(self):
        bank = build_candidate_bank(self.image, self.detections, self.config)
        allocator = TileRateAllocator()
        with torch.no_grad():
            allocator.net[-1].bias.fill_(1)
        result = compress_learned(self.image, self.detections, allocator, self.config)
        self.assertLessEqual(result.file_size, bank.target_bytes)
        self.assertTrue(result.metadata["budget_feasible"])

    def test_active_soft_budget_has_tangent_gradients(self):
        scores = torch.tensor([[0., 1., 2.], [0., 1., 2.]], dtype=torch.float64, requires_grad=True)
        costs = torch.tensor([[100., 200., 400.], [100., 300., 500.]], dtype=torch.float64)
        weights = budgeted_softmax(scores, costs, 450)
        expected = (weights * costs).sum()
        self.assertAlmostEqual(float(expected.detach()), 450, places=5)
        rate_gradient = torch.autograd.grad(expected, scores, retain_graph=True)[0]
        self.assertLess(float(rate_gradient.abs().max()), 1e-6)
        reward = (weights * torch.tensor([[0., 3., 4.], [0., 1., 2.]])).sum()
        reward.backward()
        self.assertGreater(float(scores.grad.abs().sum()), .01)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_budget_floor_fills_underspend_and_falls_back_for_discrete_gaps(self):
        costs = np.array([[100, 150, 200], [100, 150, 200]])
        scores = np.array([[2., 1., 0.], [2., 1., 0.]])
        levels = select_near_budget(scores, costs, 400)
        self.assertEqual(costs[np.arange(2), levels].sum(), 400)
        costs = np.array([[100, 300], [100, 170]])
        scores = np.array([[2., 1.], [2., 1.]])
        levels = select_near_budget(scores, costs, 400, fallback=np.array([1, 0]))
        self.assertEqual(costs[np.arange(2), levels].sum(), 400)


if __name__ == "__main__":
    unittest.main()
