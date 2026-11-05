import unittest

import torch

from nnprune.factorized_fisher import complement_update


class ComplementUpdateTest(unittest.TestCase):
    def test_matches_dense_constrained_solution(self):
        generator = torch.Generator().manual_seed(4)
        for mu in (0., 1e-4):
            for fraction in (0., .1, .5, 1.):
                with self.subTest(mu=mu, fraction=fraction):
                    weight = torch.randn(5, 8, generator=generator, dtype=torch.float64)
                    x = torch.randn(8, 8, generator=generator, dtype=weight.dtype)
                    a = x @ x.T + .1 * torch.eye(8)
                    x = torch.randn(5, 5, generator=generator, dtype=weight.dtype)
                    b = x @ x.T + .1 * torch.eye(5)
                    deleted = torch.rand(5, 8, generator=generator) < fraction
                    delta, record = complement_update(
                        weight, a, b, deleted, mu=mu, rtol=1e-9)
                    h = torch.kron(b, a)
                    free = (~deleted).flatten()
                    fixed = deleted.flatten()
                    expected = torch.zeros(weight.numel(), dtype=weight.dtype)
                    expected[fixed] = -weight.flatten()[fixed]
                    if free.any():
                        expected[free] = torch.linalg.solve(
                            h[free][:, free] + mu * torch.eye(int(free.sum())),
                            -h[free][:, fixed] @ expected[fixed])
                    torch.testing.assert_close(delta.flatten(), expected, atol=1e-7, rtol=1e-7)
                    self.assertTrue(torch.equal(delta[deleted], -weight[deleted]))
                    self.assertLessEqual(record["relative_residual"], 1e-9)

    def test_nonconvergence_is_not_applied(self):
        weight = torch.ones(2, 3, dtype=torch.float64)
        a = torch.tensor([[2., .3, .5], [.3, 1., .2], [.5, .2, 3.]], dtype=weight.dtype)
        b = torch.tensor([[2., .7], [.7, 1.]], dtype=weight.dtype)
        deleted = torch.tensor([[True, False, False], [False, True, False]])
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            complement_update(weight, a, b, deleted, rtol=1e-14, max_iterations=1)


if __name__ == "__main__":
    unittest.main()
