import unittest

import torch

from pruning.baselines.sparsegpt import sparsegpt_matrix
from pruning.methods.continuous import blockwise_obs_mask
from pruning.nnprune.factorized_fisher import complement_update


class ComplementMaskTest(unittest.TestCase):
    def test_reuses_sparsegpt_mask_and_optimizes_same_quadratic(self):
        generator = torch.Generator().manual_seed(23)
        original = torch.nn.Linear(6, 3, bias=True, dtype=torch.float64)
        with torch.no_grad():
            original.weight.copy_(torch.randn(3, 6, generator=generator,
                                              dtype=torch.float64))
        x = torch.randn(6, 6, generator=generator, dtype=torch.float64)
        a = (x.T @ x + torch.eye(6, dtype=torch.float64)).float()
        x = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        b = x.T @ x + torch.eye(3, dtype=torch.float64)
        b_inverse_diagonal = torch.linalg.inv(b).diag()
        before = original.weight.detach().clone()
        bias = original.bias.detach().clone()

        mask = blockwise_obs_mask(original, a, b_inverse_diagonal, .3, .01, 2)
        torch.testing.assert_close(original.weight, before)
        torch.testing.assert_close(original.bias, bias)

        sequential = torch.nn.Linear(6, 3, bias=False, dtype=torch.float64)
        full = torch.nn.Linear(6, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            sequential.weight.copy_(before)
            full.weight.copy_(before)
        sequential_weight, sequential_mask = sparsegpt_matrix(
            before, a, .3, percdamp=.01, blocksize=2,
            output_inverse_diagonal=b_inverse_diagonal)
        with torch.no_grad():
            sequential.weight.copy_(sequential_weight)
        self.assertTrue(torch.equal(mask, sequential_mask))

        damped = a.double().clone()
        damped.diagonal().add_(.01 * damped.diag().mean())
        delta, record = complement_update(before, damped, b, mask, rtol=1e-8, max_iterations=500)
        with torch.no_grad():
            full.weight.copy_((before + delta).masked_fill(mask, 0))
        self.assertTrue(torch.equal(full.weight == 0, mask))
        self.assertLessEqual(record["relative_residual"], 1e-8)
        damped_a = a.double() + .01 * a.double().diag().mean() * torch.eye(6, dtype=b.dtype)

        def damage(weight):
            change = weight.detach() - before
            return float(.5 * (change * (b @ change @ damped_a)).sum())

        self.assertLessEqual(damage(full.weight), damage(sequential.weight) + 1e-8)


if __name__ == "__main__":
    unittest.main()
