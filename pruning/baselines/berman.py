from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


class Berman:
    uses_selection = False

    def __init__(self, options):
        self.cfg, self.data = options, None

    def propose(self, problem, target, index, progress):
        if problem.cfg.benchmark.family == "mnist":
            return self.mnist(problem, target)
        if problem.cfg.benchmark.name == "modadd":
            return self.modadd(problem, target)
        from pruning.nnprune.fdist_baseline import fisher_loader, _sqrtF_at, _eligible, _prune_lowest
        if self.data is None:
            self.data = fisher_loader(problem.adapter, self.cfg.fisher_n, self.cfg.fisher_seed,
                                      self.cfg.batch_size)
        state, adapter = problem.state, problem.adapter
        names = _eligible(adapter, state, ())
        diagonal = _sqrtF_at(adapter, state, self.data)
        scores = {name: diagonal[name] * state.params[name].abs() for name in names}
        zeroed = sum(int((~state.masks[name]).sum()) for name in names)
        count = max(0, round(target * problem.total) - zeroed)
        _prune_lowest(state, scores, names, count)
        return {"kind": "berman", "n_deleted": count}

    def mnist(self, problem, target):
        from pruning.nnprune.third_party.fdist_fim import calculate_fim_backprop
        if self.data is None:
            x, y = problem.fisher_data(self.cfg.fisher_n, self.cfg.fisher_seed)
            self.data = DataLoader(TensorDataset(x, y), batch_size=self.cfg.batch_size)
        net = problem.model
        diagonal = calculate_fim_backprop(net, self.data, device=str(problem.device)).to(problem.device)
        parameters = list(net.named_parameters())
        weights = torch.cat([p.detach().flatten() for _, p in parameters])
        scores = diagonal.clamp_min(0).sqrt() * weights.abs()
        count = max(0, round(target * problem.total) - (problem.initial_nonzero - int(weights.count_nonzero())))
        selected = scores.masked_fill(weights == 0, torch.inf).argsort()[:count]
        weights[selected] = 0
        offset = 0
        with torch.no_grad():
            for _, parameter in parameters:
                parameter.copy_(weights[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
        return {"kind": "berman", "n_deleted": count}

    def modadd(self, problem, target):
        from pruning.nnprune.modadd_fisher import get_flat, fisher, berman_step, set_flat
        if self.data is None:
            X = problem.adapter.data["train"][0]
            generator = torch.Generator(device=X.device).manual_seed(self.cfg.fisher_seed)
            self.data = X[torch.randperm(len(X), generator=generator, device=X.device)[:self.cfg.fisher_n]]
        theta, live = get_flat(problem.adapter, problem.state)
        diagonal = fisher(problem.adapter, problem.state, self.data, blocks=False, chunk=self.cfg.batch_size)[0]
        count = max(0, round(target * problem.total) - (problem.total - int(live.sum())))
        deleted = berman_step(theta, live, diagonal, count)
        set_flat(problem.state, theta.masked_fill(deleted, 0), live & ~deleted)
        return {"kind": "berman", "n_deleted": int(deleted.sum())}
