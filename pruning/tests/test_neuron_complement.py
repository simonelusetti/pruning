import unittest

import torch
from torch import nn

from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain
from pruning.nnprune.neuron_fisher import (neuron_complement, neuron_obs_scores,
                                          select_neuron_cuts)


def spd(width):
    values = torch.randn(width, width, dtype=torch.float64)
    return values @ values.T + torch.eye(width, dtype=torch.float64)


def dense_optimum(weight, a, b, fixed, free):
    metric = torch.kron(b.contiguous(), a.contiguous())
    ids = torch.nonzero(free.flatten()).flatten()
    delta = fixed.flatten().clone()
    if len(ids):
        delta[ids] = torch.linalg.solve(metric[ids][:, ids], -(metric @ delta)[ids])
    return weight + delta.reshape_as(weight), .5 * delta @ metric @ delta


class NeuronComplementTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.incoming = torch.randn(5, 4, dtype=torch.float64)
        self.outgoing = torch.randn(3, 6, dtype=torch.float64)
        self.factors = (spd(4), spd(5), spd(6), spd(3))
        self.deleted = torch.tensor([True, False, False, False, False])
        self.incoming[self.deleted] = 0
        self.outgoing[:, :5][:, self.deleted] = 0

    def test_mixed_cuts_match_dense_quadratics_with_cleanup_reserved(self):
        inc = torch.tensor([False, True, False, False, False])
        out = torch.tensor([False, False, False, True, False])
        revised_in, revised_out, record = neuron_complement(
            self.incoming, self.outgoing, *self.factors, self.deleted, inc, out)
        survivors = ~(self.deleted | inc | out)
        free_in = survivors[:, None].expand_as(self.incoming)
        free_out = torch.cat((survivors, torch.tensor([True])))[None].expand_as(self.outgoing)
        fixed_in = torch.zeros_like(self.incoming)
        fixed_out = torch.zeros_like(self.outgoing)
        fixed_in[inc] = -self.incoming[inc]
        fixed_out[:, :5][:, out] = -self.outgoing[:, :5][:, out]
        expected_in, damage_in = dense_optimum(
            self.incoming, *self.factors[:2], fixed_in, free_in)
        expected_out, damage_out = dense_optimum(
            self.outgoing, *self.factors[2:], fixed_out, free_out)
        torch.testing.assert_close(revised_in, expected_in)
        torch.testing.assert_close(revised_out, expected_out)
        torch.testing.assert_close(revised_in[out], self.incoming[out])
        torch.testing.assert_close(revised_out[:, :5][:, inc], self.outgoing[:, :5][:, inc])
        self.assertAlmostEqual(record["predicted_fisher_damage"], float(damage_in + damage_out), places=9)
        self.assertLess(record["relative_residual"], 1e-12)
        self.assertLessEqual(record["predicted_fisher_damage"], record["no_update_fisher_damage"])

    def test_singleton_scores_equal_compensated_group_damage(self):
        scores = neuron_obs_scores(self.incoming, self.outgoing, *self.factors, self.deleted)
        self.assertTrue(bool(torch.isinf(scores[self.deleted]).all()))
        for index in torch.nonzero(~self.deleted).flatten().tolist():
            for direction in range(2):
                inc, out = torch.zeros_like(self.deleted), torch.zeros_like(self.deleted)
                (inc if direction == 0 else out)[index] = True
                _, _, record = neuron_complement(
                    self.incoming, self.outgoing, *self.factors, self.deleted, inc, out)
                self.assertAlmostEqual(float(scores[index, direction]),
                                       record["predicted_fisher_damage"], places=9)

    def test_canonicalization_preserves_primary_function_and_removes_both_sides(self):
        inc = torch.tensor([False, True, False, False, False])
        out = torch.tensor([False, False, False, True, False])
        revised_in, revised_out, _ = neuron_complement(
            self.incoming, self.outgoing, *self.factors, self.deleted, inc, out)
        fc1, fc2 = nn.Linear(3, 5, dtype=torch.float64), nn.Linear(5, 3, dtype=torch.float64)
        inputs = torch.randn(32, 3, dtype=torch.float64)
        with torch.no_grad():
            fc1.weight.copy_(revised_in[:, :-1])
            fc1.bias.copy_(revised_in[:, -1])
            fc2.weight.copy_(revised_out[:, :-1])
            fc2.bias.copy_(revised_out[:, -1])
            before = fc2(torch.relu(fc1(inputs)))
            canonicalize_linear_chain((fc1, fc2), (torch.relu,))
            after = fc2(torch.relu(fc1(inputs)))
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        removed = self.deleted | inc | out
        self.assertTrue(bool((fc1.weight[removed] == 0).all()))
        self.assertTrue(bool((fc1.bias[removed] == 0).all()))
        self.assertTrue(bool((fc2.weight[:, removed] == 0).all()))

    def test_global_selection_uses_savings_and_has_no_module_quotas(self):
        scores = [torch.tensor([[.1, 5.], [5., .2]]), torch.full((3, 2), 100.)]
        sizes = [torch.full((2,), 4), torch.full((3,), 4)]
        first, second = select_neuron_cuts(scores, sizes, 8)
        self.assertEqual(first[0].tolist(), [True, False])
        self.assertEqual(first[1].tolist(), [False, True])
        self.assertFalse(bool((second[0] | second[1]).any()))
        cuts = select_neuron_cuts([torch.tensor([[2., 4.], [1., 3.]])],
                                 [torch.tensor([10, 2])], 8)
        self.assertEqual(cuts[0][0].tolist(), [True, False])

    def test_a_neuron_cannot_be_cut_in_both_directions(self):
        mask = torch.tensor([False, True, False, False, False])
        with self.assertRaises(ValueError):
            neuron_complement(self.incoming, self.outgoing, *self.factors,
                              self.deleted, mask, mask)


if __name__ == "__main__":
    unittest.main()
