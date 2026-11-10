from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from pruning.models.opt import linear_layers
from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain
from pruning.models.mnist import linears
from pruning.curvature import activation_samples
from pruning.evaluation import paired, choose_candidate

CLASSIC_MOVES = ("unit_off", "unit_mean", "unit_merge", "head_off",
                 "head_mean", "attn_uniform", "qk_rank", "ov_rank")
ALL_MOVES = CLASSIC_MOVES + ("weight_zero",)


@dataclass
class Candidate:
    kind: str
    layer: int
    proxy: float
    savings: int
    payload: dict = field(default_factory=dict)

    @property
    def label(self):
        detail = self.payload.get("label", "")
        return f"{self.kind}:L{self.layer}" + (f":{detail}" if detail else "")


class CachedCandidateLoss:
    @torch.inference_mode()
    def __init__(self, model, sequences):
        self.model = model
        self.ids = torch.cat(sequences, 0).to(next(model.parameters()).device)
        decoder = model.model.decoder
        count = len(decoder.layers)
        self.layer_inputs = [None] * count
        self.layer_kwargs = [None] * count
        self.mlp_residuals = [None] * count
        handles = []

        for index, layer in enumerate(decoder.layers):
            def capture_layer(_module, args, kwargs, i=index):
                self.layer_inputs[i] = args[0].detach()
                self.layer_kwargs[i] = dict(kwargs)

            handles.append(layer.register_forward_pre_hook(capture_layer, with_kwargs=True))
            if layer.do_layer_norm_before:
                def capture_mlp(_module, args, i=index):
                    self.mlp_residuals[i] = args[0].detach()

                handles.append(layer.final_layer_norm.register_forward_pre_hook(capture_mlp))
        try:
            decoder(input_ids=self.ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def _finish(self, hidden, first_layer):
        decoder = self.model.model.decoder
        for index in range(first_layer, len(decoder.layers)):
            hidden = decoder.layers[index](hidden, **self.layer_kwargs[index])[0]
        if decoder.final_layer_norm is not None:
            hidden = decoder.final_layer_norm(hidden)
        if decoder.project_out is not None:
            hidden = decoder.project_out(hidden)
        logits = self.model.lm_head(hidden).contiguous()[:, :-1].float()
        labels = self.ids[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            reduction="none").reshape(self.ids.shape[0], -1).mean(1).cpu()

    @torch.inference_mode()
    def loss(self, candidate):
        layer = self.model.model.decoder.layers[candidate.layer]
        mlp_only = (candidate.kind.startswith("unit_")
                    or (candidate.kind == "weight_zero"
                        and candidate.payload["module"] in {"fc1", "fc2"}))
        residual = self.mlp_residuals[candidate.layer]
        if mlp_only and residual is not None:
            shape = self.layer_inputs[candidate.layer].shape
            hidden = layer.final_layer_norm(residual)
            hidden = layer.fc1(hidden)
            hidden = layer.activation_fn(hidden)
            hidden = layer.fc2(hidden)
            hidden = (residual + hidden).view(shape)
            return self._finish(hidden, candidate.layer + 1)
        return self._finish(self.layer_inputs[candidate.layer], candidate.layer)


def snapshot_layer(layer):
    return {name: p.detach().clone() for name, p in layer.named_parameters()}


@torch.inference_mode()
def restore_layer(layer, snapshot):
    for name, p in layer.named_parameters():
        p.copy_(snapshot[name])


def live_units(layer):
    return ((layer.fc1.weight.detach().abs().sum(1) > 0) &
            (layer.fc2.weight.detach().abs().sum(0) > 0)).cpu()


def live_heads(layer, n_heads: int):
    width = layer.self_attn.head_dim
    alive = []
    for head in range(n_heads):
        sl = slice(head * width, (head + 1) * width)
        alive.append(bool(layer.self_attn.v_proj.weight[sl].abs().sum() > 0 and
                          layer.self_attn.out_proj.weight[:, sl].abs().sum() > 0))
    return torch.tensor(alive, dtype=torch.bool)


def affected_nonzero(*tensors):
    return sum(int(torch.count_nonzero(t)) for t in tensors)


def thin_singular_values(A: torch.Tensor, B: torch.Tensor):
    QA, RA = torch.linalg.qr(A.float(), mode="reduced")
    QB, RB = torch.linalg.qr(B.float(), mode="reduced")
    return torch.linalg.svdvals(RA @ RB.T)


def lowrank_product(A: torch.Tensor, B: torch.Tensor, rank: int):
    """A' and B' with A'B'^T the rank-r truncated SVD of AB^T."""
    QA, RA = torch.linalg.qr(A.float(), mode="reduced")
    QB, RB = torch.linalg.qr(B.float(), mode="reduced")
    U, S, Vh = torch.linalg.svd(RA @ RB.T, full_matrices=False)
    root = S[:rank].sqrt()
    return QA @ (U[:, :rank] * root), QB @ (Vh[:rank].T * root)


def _top_family(candidates, kind, count):
    items = [c for c in candidates
             if c.kind == kind and c.savings > 0 and math.isfinite(c.proxy)]
    return sorted(items, key=lambda c: c.proxy)[:count]


def select_proposals(candidates, kinds, count):
    selected = []
    for kind in CLASSIC_MOVES + ("weight_zero",):
        if kind in kinds:
            selected.extend(_top_family(candidates, kind, count))
    return selected


@torch.inference_mode()
def propose(model, stats, cfg, kinds=ALL_MOVES, layer_indices=None):
    kinds = set(kinds)
    candidates = []
    hidden = model.config.hidden_size
    heads = model.config.num_attention_heads
    head_width = hidden // heads
    layers = range(len(model.model.decoder.layers)) if layer_indices is None else layer_indices

    for layer_index in layers:
        layer = model.model.decoder.layers[layer_index]
        stat = stats[layer_index]
        unit_ids = live_units(layer).nonzero().flatten().cpu()
        if unit_ids.numel() and kinds & {"unit_mean", "unit_off", "unit_merge"}:
            activations = stat.mlp_sample[:, unit_ids].float()
            columns = layer.fc2.weight[:, unit_ids].float().cpu()
            mean = stat.mlp_mean[unit_ids]
            second = stat.mlp_second[unit_ids]
            column_norm = columns.square().sum(0)
            mean_score = (second - mean.square()).clamp_min(0) * column_norm
            off_score = second * column_norm
            size = min(cfg.unit_chunk, unit_ids.numel())
            for kind, score in (("unit_mean", mean_score), ("unit_off", off_score)):
                if kind not in kinds:
                    continue
                chosen = torch.argsort(score)[:size]
                ids = unit_ids[chosen]
                savings = affected_nonzero(layer.fc1.weight[ids], layer.fc1.bias[ids],
                                           layer.fc2.weight[:, ids])
                candidates.append(Candidate(kind, layer_index, float(score[chosen].sum()),
                                            savings, {"ids": ids, "means": mean[chosen],
                                                      "label": f"n{len(ids)}"}))

            if "unit_merge" in kinds:
                pool_size = min(cfg.merge_pool, unit_ids.numel())
                pool_positions = torch.argsort(mean_score)[:pool_size]
                pool_ids = unit_ids[pool_positions]
                values = activations[:, pool_positions]
                centered = values - values.mean(0)
                scale = centered.square().mean(0).sqrt()
                normalized = centered / scale.clamp_min(1e-12)
                correlation = normalized.T @ normalized / len(normalized)
                correlation.fill_diagonal_(0)
                pairs, used = [], set()
                for flat_index in torch.argsort(correlation.abs().flatten(), descending=True).tolist():
                    source, target = divmod(flat_index, pool_size)
                    if source == target or source in used or target in used or scale[target] < 1e-12:
                        continue
                    alpha = float((centered[:, source] * centered[:, target]).mean()
                                  / centered[:, target].square().mean())
                    beta = float(values[:, source].mean() - alpha * values[:, target].mean())
                    residual = values[:, source] - alpha * values[:, target] - beta
                    proxy = float(residual.square().mean() * column_norm[pool_positions[source]])
                    pairs.append((int(pool_ids[source]), int(pool_ids[target]), alpha, beta, proxy))
                    used.update((source, target))
                    if len(pairs) == cfg.merge_chunk:
                        break
                if pairs:
                    sources = torch.tensor([pair[0] for pair in pairs])
                    savings = affected_nonzero(layer.fc1.weight[sources], layer.fc1.bias[sources],
                                               layer.fc2.weight[:, sources])
                    candidates.append(Candidate("unit_merge", layer_index,
                                                sum(pair[4] for pair in pairs), savings,
                                                {"pairs": pairs, "label": f"n{len(pairs)}"}))

        alive_heads = live_heads(layer, heads)
        for head in alive_heads.nonzero().flatten().tolist():
            section = slice(head * head_width, (head + 1) * head_width)
            common = (layer.self_attn.q_proj.weight[section],
                      layer.self_attn.q_proj.bias[section],
                      layer.self_attn.k_proj.weight[section],
                      layer.self_attn.k_proj.bias[section],
                      layer.self_attn.v_proj.weight[section],
                      layer.self_attn.v_proj.bias[section],
                      layer.self_attn.out_proj.weight[:, section])
            savings = affected_nonzero(*common)
            contribution = (stat.attn_sample[:, section].float()
                            @ layer.self_attn.out_proj.weight[:, section].float().cpu().T)
            if "head_mean" in kinds:
                centered = contribution - contribution.mean(0)
                candidates.append(Candidate("head_mean", layer_index,
                                            float(centered.square().mean()), savings,
                                            {"head": head, "mean": contribution.mean(0),
                                             "label": f"h{head}"}))
            if "head_off" in kinds:
                candidates.append(Candidate("head_off", layer_index,
                                            float(contribution.square().mean()), savings,
                                            {"head": head, "label": f"h{head}"}))

            qk = (layer.self_attn.q_proj.weight[section],
                  layer.self_attn.q_proj.bias[section],
                  layer.self_attn.k_proj.weight[section],
                  layer.self_attn.k_proj.bias[section])
            if "attn_uniform" in kinds:
                proxy = sum(float(tensor.float().square().sum()) for tensor in qk)
                candidates.append(Candidate("attn_uniform", layer_index, proxy,
                                            affected_nonzero(*qk),
                                            {"head": head, "label": f"h{head}"}))

            active_qk = ((layer.self_attn.q_proj.weight[section].abs().sum(1) > 0)
                         & (layer.self_attn.k_proj.weight[section].abs().sum(1) > 0)).nonzero().flatten()
            if "qk_rank" in kinds and active_qk.numel() > cfg.rank_chunk:
                left = torch.cat([layer.self_attn.q_proj.weight[section][active_qk].T,
                                  layer.self_attn.q_proj.bias[section][active_qk][None]], 0)
                right = layer.self_attn.k_proj.weight[section][active_qk].T
                cut = min(cfg.rank_chunk, active_qk.numel() - 1)
                rank = active_qk.numel() - cut
                proxy = float(thin_singular_values(left, right)[-cut:].square().sum())
                candidates.append(Candidate("qk_rank", layer_index, proxy,
                                            affected_nonzero(*qk) - rank * (2 * hidden + 1),
                                            {"head": head, "cut": cut,
                                             "label": f"h{head}r-{cut}"}))

            active_ov = ((layer.self_attn.v_proj.weight[section].abs().sum(1) > 0)
                         & (layer.self_attn.out_proj.weight[:, section].abs().sum(0) > 0)).nonzero().flatten()
            if "ov_rank" in kinds and active_ov.numel() > cfg.rank_chunk:
                left = torch.cat([layer.self_attn.v_proj.weight[section][active_ov].T,
                                  layer.self_attn.v_proj.bias[section][active_ov][None]], 0)
                right = layer.self_attn.out_proj.weight[:, section][:, active_ov]
                cut = min(cfg.rank_chunk, active_ov.numel() - 1)
                proxy = float(thin_singular_values(left, right)[-cut:].square().sum())
                candidates.append(Candidate("ov_rank", layer_index, proxy,
                                            cut * (2 * hidden + 1),
                                            {"head": head, "cut": cut,
                                             "label": f"h{head}r-{cut}"}))

        if "weight_zero" in kinds:
            for name, module in linear_layers(layer).items():
                live = (module.weight != 0).flatten().nonzero().flatten()
                if live.numel() == 0:
                    continue
                size = min(cfg.weight_chunk, live.numel())
                chosen = live[torch.topk(module.weight.flatten()[live].abs(),
                                         size, largest=False).indices]
                proxy = float(module.weight.flatten()[chosen].float().square().sum())
                candidates.append(Candidate("weight_zero", layer_index, proxy, size,
                                            {"module": name, "indices": chosen,
                                             "label": f"{name}.n{size}"}))

    return select_proposals(candidates, kinds, cfg.per_family)


@torch.inference_mode()
def _apply_candidate(model, candidate: Candidate):
    layer = model.model.decoder.layers[candidate.layer]
    p = candidate.payload
    if candidate.kind in ("unit_mean", "unit_off"):
        ids = p["ids"]
        if candidate.kind == "unit_mean":
            contribution = (layer.fc2.weight[:, ids].float()
                            @ p["means"].to(layer.fc2.weight).float())
            layer.fc2.bias.add_(contribution.to(layer.fc2.bias.dtype))
        # Tensor indices use advanced indexing, whose result is a copy; assign
        # through the indexed lvalue so the module is actually mutated.
        layer.fc1.weight[ids] = 0
        layer.fc1.bias[ids] = 0
        layer.fc2.weight[:, ids] = 0
        return
    if candidate.kind == "unit_merge":
        for source, target, alpha, beta, _proxy in p["pairs"]:
            column = layer.fc2.weight[:, source].clone()
            layer.fc2.weight[:, target].add_(alpha * column)
            layer.fc2.bias.add_(beta * column)
            layer.fc1.weight[source].zero_()
            layer.fc1.bias[source].zero_()
            layer.fc2.weight[:, source].zero_()
        return
    if candidate.kind in ("head_mean", "head_off", "attn_uniform", "qk_rank", "ov_rank"):
        width = layer.self_attn.head_dim
        start = p["head"] * width
        sl = slice(start, start + width)
        if candidate.kind in ("head_mean", "head_off"):
            if candidate.kind == "head_mean":
                layer.self_attn.out_proj.bias.add_(p["mean"].to(layer.self_attn.out_proj.bias))
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                               layer.self_attn.v_proj):
                projection.weight[sl].zero_(); projection.bias[sl].zero_()
            layer.self_attn.out_proj.weight[:, sl].zero_()
            return
        if candidate.kind == "attn_uniform":
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
                projection.weight[sl].zero_(); projection.bias[sl].zero_()
            return
        if candidate.kind == "qk_rank":
            q, k = layer.self_attn.q_proj, layer.self_attn.k_proj
            active = ((q.weight[sl].abs().sum(1) > 0) &
                      (k.weight[sl].abs().sum(1) > 0)).nonzero().flatten()
            rank = active.numel() - p["cut"]
            A = torch.cat([q.weight[sl][active].T, q.bias[sl][active][None]], 0)
            B = k.weight[sl][active].T
            A2, B2 = lowrank_product(A, B, rank)
            q.weight[sl].zero_(); q.bias[sl].zero_(); k.weight[sl].zero_(); k.bias[sl].zero_()
            keep = active[:rank] + start
            q.weight[keep] = A2[:-1].T.to(q.weight.dtype)
            q.bias[keep] = A2[-1].to(q.bias.dtype)
            k.weight[keep] = B2.T.to(k.weight.dtype)
            return
        v, out = layer.self_attn.v_proj, layer.self_attn.out_proj
        active = ((v.weight[sl].abs().sum(1) > 0) &
                  (out.weight[:, sl].abs().sum(0) > 0)).nonzero().flatten()
        rank = active.numel() - p["cut"]
        A = torch.cat([v.weight[sl][active].T, v.bias[sl][active][None]], 0)
        B = out.weight[:, sl][:, active]
        A2, B2 = lowrank_product(A, B, rank)
        v.weight[sl].zero_(); v.bias[sl].zero_(); out.weight[:, sl].zero_()
        keep = active[:rank] + start
        v.weight[keep] = A2[:-1].T.to(v.weight.dtype)
        v.bias[keep] = A2[-1].to(v.bias.dtype)
        out.weight[:, keep] = B2.to(out.weight.dtype)
        return
    module = dict(linear_layers(layer))[p["module"]]
    module.weight.flatten()[p["indices"]] = 0


@torch.inference_mode()
def apply_candidate(model, candidate: Candidate):
    _apply_candidate(model, candidate)
    layer = model.model.decoder.layers[candidate.layer]
    canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))


@torch.inference_mode()
def unit_classic_candidates(net, x, per_family):
    l0, l1, _ = linears(net)
    a1 = F.relu(l0(x.flatten(1)))
    acts = (a1, F.relu(l1(a1)))
    out = []
    for layer, a in enumerate(acts):
        incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
        alive = ((incoming.weight != 0).any(1) | (incoming.bias != 0)) & (outgoing.weight != 0).any(0)
        alive = alive.nonzero().flatten()
        if not alive.numel():
            continue
        a = a[:, alive]
        norms = outgoing.weight[:, alive].square().sum(0)
        means = a.mean(0)
        centered = a - means
        variance = centered.square().mean(0)
        for kind, score in (("unit_off", a.square().mean(0) * norms),
                            ("unit_mean", variance * norms)):
            for local in torch.argsort(score)[:per_family].tolist():
                c = float(means[local]) if kind == "unit_mean" else 0.
                out.append({**unit_move(kind, layer, int(alive[local]), c=c),
                            "proxy_cost": float(score[local])})
        if alive.numel() > 1:
            covariance = centered.T @ centered / len(a)
            alpha = covariance / variance.clamp_min(1e-12)[None, :]
            beta = means[:, None] - alpha * means[None, :]
            residual = variance[:, None] - covariance.square() / variance.clamp_min(1e-12)[None, :]
            cost = residual.clamp_min(0) * norms[:, None]
            cost.fill_diagonal_(torch.inf)
            for flat in torch.argsort(cost.flatten())[:per_family].tolist():
                s, t = divmod(flat, alive.numel())
                out.append({**unit_move("unit_merge", layer, int(alive[s]), int(alive[t]),
                                     float(alpha[s, t]), float(beta[s, t])),
                            "proxy_cost": float(cost[s, t])})
    return out


def unit_move(kind, layer, i, target=-1, alpha=0., c=0.):
    """unit_off: a_i -> 0.  unit_mean: a_i -> c.  unit_merge: a_i -> alpha a_target + c."""
    def apply(net):
        incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
        column = outgoing.weight[:, i].clone()
        outgoing.bias.add_(c * column)
        if target >= 0:
            outgoing.weight[:, target].add_(alpha * column)
        incoming.weight[i] = 0
        incoming.bias[i] = 0
        outgoing.weight[:, i] = 0
    label = f"{kind}:L{layer}:n{i}" + (f"->{target}" if target >= 0 else "")
    return {"kind": kind, "label": label, "apply": apply}


class OptClassic:
    uses_selection = True

    def __init__(self, options, protocol):
        self.cfg, self.protocol = options, protocol
        self.kinds = ALL_MOVES if options.weights else CLASSIC_MOVES
        self.stats, self.cache, self.dirty = None, None, None

    def propose(self, problem, target, index, progress):
        cfg, model = self.cfg, problem.model
        if (self.stats is None or index % cfg.refresh_every == 0
                or problem.cfg.sampling.refresh == "step"):
            calibration = problem.calibration[:cfg.stats_sequences] if cfg.stats_sequences else problem.calibration
            self.stats = activation_samples(model, calibration, cfg.stats_tokens, cfg.calibration_batch_size)
            self.cache, self.dirty = None, None
        if self.cache is None:
            self.cache = {i: propose(model, self.stats, cfg, self.kinds, layer_indices=(i,))
                          for i in range(problem.layers_count)}
        elif self.dirty is not None:
            self.cache[self.dirty] = propose(model, self.stats, cfg, self.kinds,
                                            layer_indices=(self.dirty,))
        proposals = select_proposals([c for items in self.cache.values() for c in items],
                                     self.kinds, cfg.per_family)
        if self.protocol.candidate_selection == "proxy":
            chosen = min(proposals, key=lambda c: c.proxy, default=None)
            if chosen is None:
                return None
            apply_candidate(model, chosen)
        else:
            loss_cache = CachedCandidateLoss(model, problem.splits["selection"])
            evaluated, snapshots = [], {}
            for candidate in proposals:
                layer = model.model.decoder.layers[candidate.layer]
                snapshots.setdefault(candidate.layer, snapshot_layer(layer))
                apply_candidate(model, candidate)
                evaluated.append((candidate, loss_cache.loss(candidate)))
                restore_layer(layer, snapshots[candidate.layer])
            if not evaluated:
                return None
            best = min(evaluated, key=lambda item: float(item[1].mean()))[1]
            ties = [item for item in evaluated
                    if paired(item[1], best)[0] <= self.protocol.n_sigma * paired(item[1], best)[1]]
            chosen, _ = max(ties, key=lambda item: (item[0].savings, -float(item[1].mean())))
            apply_candidate(model, chosen)
        self.dirty = chosen.layer
        return {"kind": chosen.kind, "move": chosen.label, "layer": chosen.layer,
                "pred_cost": chosen.proxy, "n_candidates": len(proposals)}


def fisher_weight_candidates(problem, x, per_family):
    from pruning.nnprune.network import Network
    from pruning.nnprune.fisher_metric import Fisher
    network = Network(problem, canonicalization=False)
    diagonal = Fisher(network, x).diagonal()
    proposals = []
    for module_name, module in problem.model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        name = f"{module_name}.weight" if module_name else "weight"
        start, stop = network.offsets[name]
        score = (.5 * module.weight.detach().double().flatten().square() * diagonal[start:stop])
        score.masked_fill_(module.weight.detach().flatten() == 0, torch.inf)
        for index in score.argsort()[:per_family].tolist():
            def apply(net, name=name, index=index):
                dict(net.named_parameters())[name].view(-1)[index] = 0
            proposals.append({"kind": "weight_zero", "label": f"weight_zero:{name}:{index}",
                              "apply": apply, "fisher_cost": float(score[index]),
                              "fisher_cost_per_removed": float(score[index])})
    return proposals


class MnistClassic:
    uses_selection = True

    def __init__(self, options, protocol):
        self.cfg, self.protocol = options, protocol

    def propose(self, problem, target, index, progress):
        from pruning.models.mnist import canonicalize
        x, _ = problem.fisher_data(problem.cfg.benchmark.fisher_n, problem.cfg.benchmark.fisher_seed)
        proposals = unit_classic_candidates(problem.model, x, self.cfg.per_family)
        if self.cfg.weights:
            proposals += fisher_weight_candidates(problem, x, self.cfg.per_family)
        def apply(proposal):
            with torch.no_grad():
                proposal["apply"](problem.model)
                canonicalize(problem.model)
        bound = [{**p, "apply": lambda p=p: apply(p)} for p in proposals]
        return choose_candidate(problem, bound, self.protocol)
