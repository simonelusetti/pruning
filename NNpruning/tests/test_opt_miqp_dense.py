import importlib.util
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native_baselines"))

from nnprune.factorized_fisher import inverse_diagonal
from nnprune.fisher import resolve_strategy
from run_opt_fisher_sparsegpt import fisher_layer_miqp


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "Gurobi is unavailable")
class DenseLayerMiqpTest(unittest.TestCase):
    def test_matches_enumerated_deletion_supports(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(7)
        module = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.randn(3, 4, generator=generator) * .2 + .4)
        weight = module.weight.detach().double().clone()
        x = torch.randn(4, 4, generator=generator, dtype=torch.float64)
        a = x @ x.T + torch.eye(4, dtype=torch.float64)
        x = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        b = x @ x.T + torch.eye(3, dtype=torch.float64)
        a_inverse = inverse_diagonal(a)
        b_inverse = inverse_diagonal(b)

        saliency = weight.square() / (b_inverse[:, None] * a_inverse[None, :])
        deletion_ids = torch.topk(saliency.flatten(), 4, largest=False).indices
        rows, columns = deletion_ids // 4, deletion_ids % 4
        correlation = ((b[:, rows] * weight.flatten()[deletion_ids][None]) @
                       a[:, columns].T)
        score = correlation.abs() / (b.diag()[:, None] * a.diag()[None]).sqrt()
        score.flatten()[deletion_ids] = -1
        update_ids = torch.topk(score.flatten(), 4).indices
        coordinates = torch.cat((deletion_ids, update_ids))
        rows, columns = coordinates // 4, coordinates % 4
        q = b[rows[:, None], rows[None]] * a[columns[:, None], columns[None]]

        minimum = float("inf")
        for support in itertools.combinations(range(4), 2):
            deletion_delta = torch.zeros(4, dtype=torch.float64)
            deletion_delta[list(support)] = -weight.flatten()[deletion_ids[list(support)]]
            update_delta = -torch.linalg.solve(
                q[4:, 4:], q[4:, :4] @ deletion_delta)
            delta = torch.cat((deletion_delta, update_delta))
            minimum = min(minimum, float(.5 * delta @ q @ delta))

        args = SimpleNamespace(
            percdamp=.01, fisher_pool=4, fisher_mu=0.,
            fisher_lambda_u=0., fisher_lambda_d=0., fisher_bound=10.,
            fisher_time_limit=10., fisher_gap=0., fisher_log=False,
            fisher_strategy=resolve_strategy(
                "fisher_sparsegpt", geometry="axb", deletion="miqp",
                compensation="dense", update_scope="layer",
                structure="none", decision="proxy"),
            _a_inverse_diagonal=a_inverse)
        record = fisher_layer_miqp(
            module, a, b, b_inverse, .1, args, prepared=True, deletion_count=2)
        self.assertAlmostEqual(record["predicted_fisher_damage"], minimum, places=9)
        self.assertEqual(record["updates"], 4)
        self.assertEqual(record["deleted"], 2)


if __name__ == "__main__":
    unittest.main()
