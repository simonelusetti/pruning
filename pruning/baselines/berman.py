from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


class Berman:
    uses_selection = False

    def __init__(self, options):
        self.cfg, self.data = options, None
        self.scheme = options.get("scheme", "iterative")
        self.cached_scores = None
        if self.scheme not in {"one_shot", "iterative", "global", "exact", "magnitude", "fim"} or options.get("avg_points", 3) < 2:
            raise ValueError("Invalid Berman scheme or path quadrature size")

    def propose(self, problem, target, index, progress):
        if self.scheme != "iterative" or self.cfg.get("exclude"):
            return self.path_variant(problem, target)
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

    def path_variant(self, problem, target):
        from pruning.nnprune.fdist_baseline import AdapterModule, fisher_loader, _eligible, _prune_lowest
        from pruning.nnprune.third_party.fdist_fim import calculate_fim_backprop, fisher_entries_forward
        if hasattr(problem, "model"):
            net, device = problem.model, problem.device
            if self.data is None:
                self.data = DataLoader(TensorDataset(*problem.fisher_data(self.cfg.fisher_n, self.cfg.fisher_seed)),
                                       batch_size=self.cfg.batch_size)
        else:
            net = AdapterModule(problem.adapter, problem.state)
            device = next(net.parameters()).device
            if self.data is None:
                self.data = fisher_loader(problem.adapter, self.cfg.fisher_n, self.cfg.fisher_seed, self.cfg.batch_size)
        parameters = dict(net.named_parameters())
        excluded = set(self.cfg.get("exclude", ()))
        original = {name: value.detach().clone() for name, value in parameters.items()}
        def diagonal():
            flat = calculate_fim_backprop(net, self.data, device=str(device)).to(device)
            output, offset = {}, 0
            for name, value in parameters.items():
                output[name] = flat[offset:offset+value.numel()].view_as(value)
                offset += value.numel()
            return output
        if self.scheme in {"one_shot", "magnitude", "fim"} and self.cached_scores is not None:
            scores = self.cached_scores
        elif self.scheme == "magnitude":
            scores = {name: value.abs() for name, value in original.items()}
            self.cached_scores = scores
        else:
            points = self.cfg.get("avg_points", 3)
            if self.scheme == "global":
                roots = {name: torch.zeros_like(value) for name, value in original.items()}
                try:
                    for alpha in torch.linspace(1., 1./points, points).tolist():
                        with torch.no_grad():
                            for name, value in parameters.items():
                                value.copy_(original[name] * alpha)
                        for name, value in diagonal().items():
                            roots[name] += value.clamp_min(0).sqrt() / points
                finally:
                    with torch.no_grad():
                        for name, value in parameters.items():
                            value.copy_(original[name])
            else:
                metric = diagonal()
                if self.scheme == "exact":
                    for name, value in original.items():
                        if name in excluded:
                            continue
                        indices = value.flatten().ne(0).nonzero().flatten()
                        total = metric[name].flatten()[indices].clone()
                        for alpha in torch.linspace(1., 0., points).tolist()[1:]:
                            total += fisher_entries_forward(net, self.data, name, indices,
                                torch.full((len(indices),), alpha, device=device), device=str(device),
                                probe_batch=self.cfg.get("probe_batch", 32), sample_chunk=self.cfg.get("sample_chunk", 16)).to(device)
                        metric[name].flatten()[indices] = total / points
                roots = {name: value.clamp_min(0).sqrt() for name, value in metric.items()}
            scores = metric if self.scheme == "fim" else {name: roots[name] * value.abs() for name, value in original.items()}
            if self.scheme in {"one_shot", "fim"}:
                self.cached_scores = scores
        if hasattr(problem, "model"):
            weights = torch.cat([value.flatten() for value in original.values()])
            score = torch.cat([scores[name].flatten() for name in original])
            allowed = torch.cat([torch.full_like(value, name not in excluded, dtype=torch.bool).flatten() for name, value in original.items()])
            count = max(0, round(target * problem.total) - (problem.initial_nonzero - int(weights.count_nonzero())))
            selected = score.masked_fill(weights.eq(0) | ~allowed, torch.inf).topk(min(count, int((weights.ne(0) & allowed).sum())), largest=False).indices
            count = len(selected)
            weights[selected] = 0
            offset = 0
            with torch.no_grad():
                for value in parameters.values():
                    value.copy_(weights[offset:offset+value.numel()].view_as(value))
                    offset += value.numel()
        else:
            names = _eligible(problem.adapter, problem.state, excluded)
            zeroed = sum(int((~problem.state.masks[name]).sum()) for name in names)
            count = _prune_lowest(problem.state, scores, names, max(0, round(target * problem.total) - zeroed))
        return {"kind": f"berman_{self.scheme}", "n_deleted": count}

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
