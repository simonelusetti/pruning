import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native_baselines"))
from run_compensation_ablation import ALPHAS, apply_plan, exact_fisher_plan, predicted


class CompensationAblationTest(unittest.TestCase):
    def test_exact_fisher_includes_cross_layer_interactions(self):
        generator = torch.Generator().manual_seed(12)
        plans, activations, gradients, columns = [], [], [], []
        for rows, width in ((2, 3), (2, 2)):
            weight = torch.randn(rows, width, generator=generator, dtype=torch.float64)
            deleted = torch.rand(rows, width, generator=generator) < .4
            a = torch.randn(7, width + 1, generator=generator, dtype=torch.float64)
            g = torch.randn(7, 3, rows, generator=generator, dtype=torch.float64)
            plans.append((weight, deleted, torch.zeros_like(weight)))
            activations.append(a)
            gradients.append(g)
            columns.append((g[:, :, :, None] * a[:, None, None, :-1]).reshape(21, -1))
        design = torch.cat(columns, 1) / (7 ** .5)
        hessian = design.T @ design
        mask = torch.cat([d.flatten() for _, d, _ in plans])
        fixed = torch.cat([torch.where(d, -w, 0.).flatten() for w, d, _ in plans])
        free = ~mask
        mu = .01
        expected = torch.linalg.solve(
            hessian[free][:, free] + mu * torch.eye(int(free.sum())),
            -(hessian @ fixed)[free])
        solution, record = exact_fisher_plan(
            plans, activations, gradients, SimpleNamespace(mu=mu))
        update = torch.cat([u.flatten() for _, _, u in solution])
        torch.testing.assert_close(update[free], expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(update[mask], torch.zeros_like(update[mask])))
        self.assertLessEqual(record["relative_residual"], 1e-6)
        self.assertLessEqual(predicted(record["coefficients"], 1)[1],
                             predicted(record["coefficients"], 0)[1])

    def test_interpolation_preserves_the_mask_and_original_weights(self):
        layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        weight = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
        mask = torch.tensor([[True, False, False], [False, True, False]])
        update = torch.full_like(weight, .1).masked_fill(mask, 0.)
        for alpha in ALPHAS:
            apply_plan([layer], [(weight, mask, update)], alpha)
            self.assertTrue(torch.equal(layer.weight[mask], torch.zeros(2, dtype=weight.dtype)))
            torch.testing.assert_close(layer.weight[~mask], (weight + alpha * update)[~mask])
        self.assertEqual(float(weight.sum()), 21.)


if __name__ == "__main__":
    unittest.main()
