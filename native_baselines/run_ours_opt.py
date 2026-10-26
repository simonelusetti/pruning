#!/usr/bin/env python3
"""CPU pilot of the project's greedy predictive framework on Hugging Face OPT.

The adapter uses the moves that can be represented faithfully inside an
ordinary OPT decoder block: MLP unit zero/mean/merge, head zero/mean, uniform
attention, QK/OV rank cuts, and weight-zero chunks.  Candidate models are
applied in-place and rolled back, avoiding a full model clone per candidate.

This is a scaled CPU protocol, not the toy-GPT protocol or a paper result.
Move chunks are enlarged in proportion to model size and validation subsets are
small enough to make greedy candidate evaluation feasible without CUDA.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))
from nnprune.canonicalize.mlp import canonicalize_linear_chain
from run_cpu_close import (DATASET_CACHE, calibration_tokens, linear_layers,
                           official_test_tokens, per_sequence_loss, perplexity,
                           selection_sequences,
                           wikitext_test_tokens)
from result_artifacts import write_json

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


@dataclass
class ActivationStats:
    mlp_sample: torch.Tensor
    attn_sample: torch.Tensor
    mlp_mean: torch.Tensor
    mlp_second: torch.Tensor


def validation_sequences(tokenizer, seqlen: int, n_a: int, n_b: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    need = (n_a + n_b) * seqlen
    if ids.numel() < need:
        raise ValueError(f"validation split has {ids.numel()} tokens, need {need}")
    chunks = [ids[i:i + seqlen].unsqueeze(0) for i in range(0, need, seqlen)]
    return chunks[:n_a], chunks[n_a:]


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

def paired(mean_candidate: torch.Tensor, reference: torch.Tensor):
    delta = (mean_candidate - reference).double()
    mean = float(delta.mean())
    se = (float(delta.std(unbiased=True)) / math.sqrt(delta.numel())
          if delta.numel() > 1 else float("inf"))
    return mean, se


def decoder_linear_parameters(model):
    for layer in model.model.decoder.layers:
        for module in linear_layers(layer).values():
            yield module.weight
            if module.bias is not None:
                yield module.bias


def nonzero_count(model):
    return sum(int(torch.count_nonzero(p)) for p in decoder_linear_parameters(model))


def total_count(model):
    return sum(p.numel() for p in decoder_linear_parameters(model))


def snapshot_layer(layer):
    return {name: p.detach().clone() for name, p in layer.named_parameters()}


@torch.inference_mode()
def restore_layer(layer, snapshot):
    for name, p in layer.named_parameters():
        p.copy_(snapshot[name])


@torch.inference_mode()
def activation_samples(model, sequences, max_tokens: int, batch_size: int):
    """Summarise every calibration sequence and retain a distributed sample.

    MLP means and second moments use every token.  Correlation-heavy merge and
    attention proposals use at most ``max_tokens`` positions, sampled evenly
    from every calibration document instead of taking only the first documents.
    """
    layers = model.model.decoder.layers
    mlp, attn = [[] for _ in layers], [[] for _ in layers]
    mlp_sum = [None for _ in layers]
    mlp_sumsq = [None for _ in layers]
    mlp_count = [0 for _ in layers]
    quota = max(1, math.ceil(max_tokens / len(sequences)))
    handles = []

    def distributed_sample(value, have):
        batch = value.shape[0] if value.ndim > 2 else 1
        value = value.detach().reshape(batch, -1, value.shape[-1]).float().cpu()
        take = min(quota, value.shape[1])
        positions = torch.linspace(0, value.shape[1] - 1, take).long()
        return value[:, positions].reshape(-1, value.shape[-1])[:max_tokens - have]

    for li, layer in enumerate(layers):
        def capture_mlp(_module, args, index=li):
            nonlocal mlp_sum, mlp_sumsq, mlp_count
            x_all = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu()
            total = x_all.sum(0)
            total_sq = x_all.square().sum(0)
            mlp_sum[index] = total if mlp_sum[index] is None else mlp_sum[index] + total
            mlp_sumsq[index] = (total_sq if mlp_sumsq[index] is None
                                else mlp_sumsq[index] + total_sq)
            mlp_count[index] += x_all.shape[0]
            have = sum(x.shape[0] for x in mlp[index])
            if have < max_tokens:
                mlp[index].append(distributed_sample(args[0], have))

        def capture_attn(_module, args, index=li):
            have = sum(x.shape[0] for x in attn[index])
            if have < max_tokens:
                attn[index].append(distributed_sample(args[0], have))

        handles.append(layer.fc2.register_forward_pre_hook(capture_mlp))
        handles.append(layer.self_attn.out_proj.register_forward_pre_hook(capture_attn))
    try:
        device = next(model.parameters()).device
        dataset = TensorDataset(torch.cat(sequences, 0))
        for (ids,) in DataLoader(dataset, batch_size=batch_size):
            model.model.decoder(ids.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [ActivationStats(
        torch.cat(mlp[i], 0), torch.cat(attn[i], 0),
        mlp_sum[i] / mlp_count[i], mlp_sumsq[i] / mlp_count[i])
        for i in range(len(layers))]


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
def propose(model, stats, args, kinds=ALL_MOVES, layer_indices=None):
    kinds = set(kinds)
    candidates = []
    hidden = model.config.hidden_size
    heads = model.config.num_attention_heads
    head_width = hidden // heads
    layers = range(len(model.model.decoder.layers)) if layer_indices is None else layer_indices

    for layer_index in layers:
        layer = model.model.decoder.layers[layer_index]
        stat = stats[layer_index]
        unit_ids = live_units(layer).nonzero().flatten()
        if unit_ids.numel() and kinds & {"unit_mean", "unit_off", "unit_merge"}:
            activations = stat.mlp_sample[:, unit_ids].float()
            columns = layer.fc2.weight[:, unit_ids].float().cpu()
            mean = stat.mlp_mean[unit_ids]
            second = stat.mlp_second[unit_ids]
            column_norm = columns.square().sum(0)
            mean_score = (second - mean.square()).clamp_min(0) * column_norm
            off_score = second * column_norm
            size = min(args.unit_chunk, unit_ids.numel())
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
                pool_size = min(args.merge_pool, unit_ids.numel())
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
                    if len(pairs) == args.merge_chunk:
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
            if "qk_rank" in kinds and active_qk.numel() > args.rank_chunk:
                left = torch.cat([layer.self_attn.q_proj.weight[section][active_qk].T,
                                  layer.self_attn.q_proj.bias[section][active_qk][None]], 0)
                right = layer.self_attn.k_proj.weight[section][active_qk].T
                cut = min(args.rank_chunk, active_qk.numel() - 1)
                rank = active_qk.numel() - cut
                proxy = float(thin_singular_values(left, right)[-cut:].square().sum())
                candidates.append(Candidate("qk_rank", layer_index, proxy,
                                            affected_nonzero(*qk) - rank * (2 * hidden + 1),
                                            {"head": head, "cut": cut,
                                             "label": f"h{head}r-{cut}"}))

            active_ov = ((layer.self_attn.v_proj.weight[section].abs().sum(1) > 0)
                         & (layer.self_attn.out_proj.weight[:, section].abs().sum(0) > 0)).nonzero().flatten()
            if "ov_rank" in kinds and active_ov.numel() > args.rank_chunk:
                left = torch.cat([layer.self_attn.v_proj.weight[section][active_ov].T,
                                  layer.self_attn.v_proj.bias[section][active_ov][None]], 0)
                right = layer.self_attn.out_proj.weight[:, section][:, active_ov]
                cut = min(args.rank_chunk, active_ov.numel() - 1)
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
                size = min(args.weight_chunk, live.numel())
                chosen = live[torch.topk(module.weight.flatten()[live].abs(),
                                         size, largest=False).indices]
                proxy = float(module.weight.flatten()[chosen].float().square().sum())
                candidates.append(Candidate("weight_zero", layer_index, proxy, size,
                                            {"module": name, "indices": chosen,
                                             "label": f"{name}.n{size}"}))

    return select_proposals(candidates, kinds, args.per_family)


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
                layer.self_attn.out_proj.bias.add_(p["mean"].to(layer.self_attn.out_proj.bias.dtype))
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--val-a", type=int, default=2)
    p.add_argument("--val-b", type=int, default=4)
    p.add_argument("--selection-dataset", choices=("wikitext2", "c4"),
                   default="wikitext2")
    p.add_argument("--selection-seed", type=int, default=12345)
    p.add_argument("--stats-tokens", type=int, default=1024)
    p.add_argument("--stats-sequences", type=int, default=0,
                   help="calibration sequences used for proxies; zero uses all")
    p.add_argument("--calibration-batch-size", type=int, default=4)
    p.add_argument("--move-group", choices=("classic", "classic_weights"),
                   default="classic")
    p.add_argument("--eval-tokens", type=int, default=32768)
    p.add_argument("--max-steps", type=int, default=0,
                   help="optional safety cap; zero runs until the stopping rule")
    p.add_argument("--refresh-every", type=int, default=5)
    p.add_argument("--unit-chunk", type=int, default=32,
                   help="units removed by each classic off/mean proposal")
    p.add_argument("--merge-pool", type=int, default=128)
    p.add_argument("--merge-chunk", type=int, default=32)
    p.add_argument("--rank-chunk", type=int, default=8)
    p.add_argument("--weight-chunk", type=int, default=32768)
    p.add_argument("--per-family", type=int, default=3)
    p.add_argument("--rho", type=float, default=.05)
    p.add_argument("--n-sigma", type=float, default=2.)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path,
                   default=ROOT / "native_baselines/results_cpu_close/ours_opt.json")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "native_baselines/results_cpu_close/ours_opt_checkpoint.json")
    p.add_argument("--loss-targets", type=Path,
                   help="baseline endpoint losses; when set, continue until all are crossed")
    p.add_argument("--model-checkpoint", type=Path,
                   default=ROOT / "native_baselines/results_cpu_close/ours_opt_model.pt")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--no-model-save", action="store_true")
    p.add_argument("--published-opt", action="store_true")
    p.add_argument("--skip-test-evaluation", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-after-stop", action="store_true")
    args = p.parse_args()
    if args.rho < 0 or args.n_sigma < 0:
        p.error("--rho and --n-sigma must be non-negative")
    if min(args.nsamples, args.seqlen, args.val_a, args.val_b,
           args.stats_tokens, args.refresh_every, args.per_family,
           args.calibration_batch_size) < 1:
        p.error("sample, sequence, refresh, and candidate counts must be positive")
    if args.max_steps < 0:
        p.error("--max-steps must be non-negative")
    if args.stats_sequences < 0 or args.stats_sequences > args.nsamples:
        p.error("--stats-sequences must be between zero and --nsamples")
    if args.published_opt and not (args.model == "facebook/opt-125m"
                                   and args.nsamples == 128
                                   and args.seqlen == 2048 and args.seed == 0):
        p.error("published OPT mode requires OPT-125M, 128x2048 C4 samples, seed 0")
    return args


def write_checkpoint(args, rows, status, signature):
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    temporary.write_text(json.dumps(
        {"result_schema": 2, "status": status, "signature": signature,
         "rows": rows}, indent=2) + "\n")
    os.replace(temporary, args.checkpoint)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=not args.published_opt)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=("auto" if args.published_opt else torch.float32)).to(device).eval()
    model.config.use_cache = False
    calibration = calibration_tokens(tokenizer, "c4", args.nsamples, args.seqlen, args.seed)
    if args.selection_dataset == "wikitext2":
        val_a, val_b = validation_sequences(
            tokenizer, args.seqlen, args.val_a, args.val_b)
    else:
        selected = selection_sequences(
            tokenizer, "c4", args.seqlen, 0, args.val_a + args.val_b,
            args.selection_seed)
        val_a, val_b = selected[:args.val_a], selected[args.val_a:]
    ref_a, ref_b = per_sequence_loss(model, val_a), per_sequence_loss(model, val_b)
    base_loss = float(ref_b.mean())
    base_nonzero = nonzero_count(model)
    prunable_total = total_count(model)
    model_total = sum(parameter.numel() for parameter in model.parameters())
    signature = {
        "model": args.model, "move_group": args.move_group,
        "seed": args.seed, "nsamples": args.nsamples, "seqlen": args.seqlen,
        "selection_dataset": args.selection_dataset,
        "selection_seed": args.selection_seed, "val_a": args.val_a,
        "val_b": args.val_b, "rho": args.rho, "n_sigma": args.n_sigma,
        "stats_tokens": args.stats_tokens,
        "stats_sequences": args.stats_sequences,
        "calibration_batch_size": args.calibration_batch_size,
        "refresh_every": args.refresh_every, "unit_chunk": args.unit_chunk,
        "merge_pool": args.merge_pool, "merge_chunk": args.merge_chunk,
        "rank_chunk": args.rank_chunk, "weight_chunk": args.weight_chunk,
        "per_family": args.per_family,
    }
    targets = {}
    if args.loss_targets:
        target_data = json.loads(args.loss_targets.read_text())
        if "methods" in target_data:
            targets = {name: float(info["mean_loss"])
                       for name, info in target_data["methods"].items()
                       if name != "dense"}
        elif target_data.get("selection_loss") is not None:
            targets = {target_data.get("method", "baseline"):
                       float(target_data["selection_loss"])}
        else:
            raise ValueError("loss-target file has no usable selection loss")
    rows = [{"step": 0, "kind": "", "move": "", "removed": 0,
             "frac_removed": 0., "val_a_loss": float(ref_a.mean()),
             "val_b_loss": float(ref_b.mean()), "val_b_delta": 0.,
             "val_b_se": 0., "matched_baselines": [],
             "seconds": time.perf_counter() - started}]
    if args.resume and not args.no_model_save and args.model_checkpoint.exists():
        saved = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError("checkpoint settings do not match this run")
        model.load_state_dict(saved["model"])
        rows = saved["rows"]
        print(f"RESUME after step {rows[-1]['step']}", flush=True)
    write_checkpoint(args, rows, "running", signature)
    kinds = {"classic": CLASSIC_MOVES, "classic_weights": ALL_MOVES}[args.move_group]
    stats, proposal_cache = None, None
    dirty_layer = None
    completed_targets = set(rows[-1].get("matched_baselines", []))
    stopped = any(not row.get("admissible", True) for row in rows)
    stop_index = next((i - 1 for i, row in enumerate(rows)
                       if not row.get("admissible", True)), None)
    first_rejected = next((row for row in rows if not row.get("admissible", True)), None)
    rejected_candidate = (dict(first_rejected) if first_rejected is not None else None)
    termination_reason = None
    if stopped and not args.continue_after_stop:
        termination_reason = "stop_rule"
    first_step = int(rows[-1]["step"]) + 1
    step_iterator = (range(first_step, args.max_steps + 1) if args.max_steps > 0
                     else itertools.count(first_step))
    for step in (() if termination_reason is not None else step_iterator):
        if stats is None or (step - 1) % args.refresh_every == 0:
            print(f"refreshing activation statistics at step {step}", flush=True)
            stats_calibration = (calibration[:args.stats_sequences]
                                 if args.stats_sequences else calibration)
            stats = activation_samples(model, stats_calibration, args.stats_tokens,
                                       args.calibration_batch_size)
            proposal_cache = None
            dirty_layer = None
        if proposal_cache is None:
            proposal_cache = {
                layer: propose(model, stats, args, kinds, layer_indices=(layer,))
                for layer in range(len(model.model.decoder.layers))
            }
        elif dirty_layer is not None:
            proposal_cache[dirty_layer] = propose(
                model, stats, args, kinds, layer_indices=(dirty_layer,))
            dirty_layer = None
        candidates = select_proposals(
            [candidate for layer in range(len(model.model.decoder.layers))
             for candidate in proposal_cache[layer]], kinds, args.per_family)
        evaluated = []
        snapshots = {}
        loss_cache = CachedCandidateLoss(model, val_a)
        for candidate in candidates:
            layer = model.model.decoder.layers[candidate.layer]
            if candidate.layer not in snapshots:
                snapshots[candidate.layer] = snapshot_layer(layer)
            apply_candidate(model, candidate)
            loss = loss_cache.loss(candidate)
            evaluated.append((candidate, loss))
            restore_layer(layer, snapshots[candidate.layer])
        if not evaluated:
            termination_reason = "no_candidates"
            break
        best_loss = min(float(loss.mean()) for _, loss in evaluated)
        ties = []
        best_vector = min(evaluated, key=lambda x: float(x[1].mean()))[1]
        for candidate, loss in evaluated:
            difference = (loss - best_vector).double()
            se = (float(difference.std(unbiased=True)) / math.sqrt(len(difference))
                  if len(difference) > 1 else float("inf"))
            if float(difference.mean()) <= args.n_sigma * se:
                ties.append((candidate, loss))
        chosen, loss_a = max(ties, key=lambda x: (x[0].savings, -float(x[1].mean())))
        layer = model.model.decoder.layers[chosen.layer]
        snapshot = snapshot_layer(layer)
        apply_candidate(model, chosen)
        removed = base_nonzero - nonzero_count(model)
        if removed <= rows[-1]["removed"]:
            restore_layer(layer, snapshot)
            raise RuntimeError(f"candidate made no pruning progress: {chosen.label}")
        loss_b = per_sequence_loss(model, val_b)
        delta, se = paired(loss_b, ref_b)
        passes_rule = (True if targets else
                       delta <= args.n_sigma * se + args.rho * base_loss)
        if not stopped and not passes_rule:
            print(f"STOP step {step}: {chosen.label} fails anchored valB "
                  f"({delta:.4g} > {args.n_sigma * se + args.rho * base_loss:.4g})", flush=True)
            stop_index = len(rows) - 1
            stopped = True
            rejected_candidate = {
                "step": step, "kind": chosen.kind, "move": chosen.label,
                "removed": base_nonzero - nonzero_count(model),
                "val_b_loss": float(loss_b.mean()), "val_b_delta": delta,
                "val_b_se": se,
                "relative_damage": float(loss_b.mean()) / float(ref_b.mean()) - 1,
                "relative_bound": args.rho + args.n_sigma * se / base_loss,
            }
            if not args.continue_after_stop:
                restore_layer(layer, snapshot)
                termination_reason = "stop_rule"
                break
        mean_b = float(loss_b.mean())
        completed_targets |= {name for name, target in targets.items() if mean_b >= target}
        row = {"step": step, "kind": chosen.kind, "move": chosen.label,
               "removed": removed, "frac_removed": removed / prunable_total,
               "val_a_loss": float(loss_a.mean()), "val_b_loss": mean_b,
               "val_b_delta": delta, "val_b_se": se,
               "admissible": not stopped,
               "matched_baselines": sorted(completed_targets),
               "seconds": time.perf_counter() - started}
        rows.append(row)
        write_checkpoint(args, rows, "running", signature)
        print(f"step {step}: {chosen.label}, removed {100 * removed / prunable_total:.3f}%, "
              f"valB dL={delta:.4g}, {row['seconds']:.0f}s", flush=True)
        if step % args.refresh_every == 0:
            stats = None
            proposal_cache = None
        else:
            dirty_layer = chosen.layer
        if not args.no_model_save and step % args.save_every == 0:
            tmp = args.model_checkpoint.with_suffix(args.model_checkpoint.suffix + ".tmp")
            torch.save({"model": model.state_dict(), "rows": rows,
                        "signature": signature}, tmp)
            os.replace(tmp, args.model_checkpoint)
        if targets and completed_targets == set(targets):
            print(f"TARGETS REACHED at step {step}: {sorted(completed_targets)}", flush=True)
            termination_reason = "loss_targets"
            break
    if termination_reason is None:
        termination_reason = "max_steps"

    if args.skip_test_evaluation:
        evaluations, ppl, labels, eval_seconds = {}, float("nan"), 0, 0.
    elif args.published_opt:
        evaluations = {}
        eval_seconds = 0.
        for dataset in ("wikitext2", "ptb", "c4"):
            test_ids = official_test_tokens(tokenizer, dataset, args.seqlen)
            value, count, seconds = perplexity(model, test_ids.to(device), args.seqlen)
            evaluations[dataset] = {"perplexity": value, "evaluated_labels": count,
                                    "seconds": seconds}
            eval_seconds += seconds
        ppl = evaluations["wikitext2"]["perplexity"]
        labels = evaluations["wikitext2"]["evaluated_labels"]
    else:
        test_ids = wikitext_test_tokens(tokenizer, args.eval_tokens)
        ppl, labels, eval_seconds = perplexity(model, test_ids.to(device), args.seqlen)
        evaluations = {"wikitext2": {"perplexity": ppl,
                                       "evaluated_labels": labels,
                                       "seconds": eval_seconds}}
    endpoint_index = stop_index if stop_index is not None else len(rows) - 1
    endpoint = rows[endpoint_index]
    final_point = rows[-1]
    result = {"result_schema": 2, "status": "complete",
              "kind": "our greedy framework, CPU-close OPT port",
              "model": args.model, "move_group": args.move_group,
              "protocol": ("published SparseGPT OPT model/calibration/evaluation; "
                           f"framework selection uses an additional "
                           f"{args.selection_dataset} split"
                           if args.published_opt else "CPU-close OPT pilot"),
              "moves": list(kinds),
              "calibration": {"dataset": "c4", "nsamples": args.nsamples,
                              "seqlen": args.seqlen, "stats_tokens": args.stats_tokens,
                              "stats_sequences": (args.stats_sequences or args.nsamples)},
              "selection": {"dataset": args.selection_dataset,
                            "seed": args.selection_seed, "val_a": args.val_a,
                            "val_b": args.val_b, "rho": args.rho,
                            "n_sigma": args.n_sigma},
              "candidate_budget": {"unit_chunk": args.unit_chunk,
                                   "merge_pool": args.merge_pool,
                                   "merge_chunk": args.merge_chunk,
                                   "rank_chunk": args.rank_chunk,
                                   "per_family": args.per_family},
              "loss_targets": targets,
              "termination_reason": termination_reason,
              "stop_index": stop_index,
              "stopping_point": rows[stop_index] if stop_index is not None else None,
              "accepted_endpoint": endpoint,
              "rejected_candidate": rejected_candidate,
              "accepted_steps": endpoint["step"],
              "removed_parameters": endpoint["removed"],
              "frac_removed": endpoint["frac_removed"],
              "history_steps": len(rows) - 1,
              "final_history_point": final_point,
              "test_evaluation_point": "final_history",
              "wikitext2_perplexity_at_final_history": ppl,
              "evaluations": evaluations,
              "prunable_decoder_linear_parameters": prunable_total,
              "total_model_parameters": model_total,
              "evaluated_labels": labels,
              "eval_seconds": eval_seconds, "total_seconds": time.perf_counter() - started,
              "history": rows}
    write_json(args.output, result, "opt")
    if not args.no_model_save:
        tmp = args.model_checkpoint.with_suffix(args.model_checkpoint.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "rows": rows,
                    "signature": signature}, tmp)
        os.replace(tmp, args.model_checkpoint)
    write_checkpoint(args, rows, "complete", signature)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
