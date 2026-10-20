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
from nnprune.mlp import (grouped_fisher_compensation,
                         grouped_fisher_pair_compensations,
                         quantization_metadata,
                         quantize_grouped_fisher_compensation)

from run_cpu_close import (DATASET_CACHE, calibration_tokens, linear_layers,
                           official_test_tokens, perplexity, selection_sequences,
                           wikitext_test_tokens)

CLASSIC_MOVES = ("unit_off", "unit_mean", "unit_merge", "head_off",
                 "head_mean", "attn_uniform", "qk_rank", "ov_rank")
FISHER_SHARED_MOVES = ("attn_uniform", "qk_rank", "ov_rank")
FISHER_SINGLE_MOVES = ("unit_fisher", "head_fisher") + FISHER_SHARED_MOVES
FISHER_MOVES = (("unit_fisher", "head_fisher", "unit_fisher_pair",
                 "head_fisher_pair") + FISHER_SHARED_MOVES)
FISHER_QUANTIZED_MOVES = (("unit_fisher_quantized", "head_fisher_quantized")
                          + FISHER_SHARED_MOVES)
FISHER_ALL_MOVES = ("unit_fisher", "head_fisher", "unit_fisher_pair",
                    "head_fisher_pair", "unit_fisher_quantized",
                    "head_fisher_quantized")
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


@dataclass
class FisherStats:
    mlp_sample: torch.Tensor
    mlp_grad: torch.Tensor
    attn_sample: torch.Tensor
    attn_grad: torch.Tensor


def validation_sequences(tokenizer, seqlen: int, n_a: int, n_b: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    need = (n_a + n_b) * seqlen
    if ids.numel() < need:
        raise ValueError(f"validation split has {ids.numel()} tokens, need {need}")
    chunks = [ids[i:i + seqlen].unsqueeze(0) for i in range(0, need, seqlen)]
    return chunks[:n_a], chunks[n_a:]


@torch.inference_mode()
def per_sequence_loss(model, sequences: list[torch.Tensor]) -> torch.Tensor:
    ids = torch.cat(sequences, 0).to(next(model.parameters()).device)
    logits = model(ids, use_cache=False).logits[:, :-1].float()
    labels = ids[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none").reshape(ids.shape[0], -1).mean(1)
    return losses.cpu()


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


def fisher_samples(model, sequences, max_tokens: int, n_mc: int, seed: int):
    layers = model.model.decoder.layers
    mlp, mlp_grad = [[] for _ in layers], [[] for _ in layers]
    attn, attn_grad = [[] for _ in layers], [[] for _ in layers]
    current_mlp, current_attn = {}, {}
    current_mlp_out, current_attn_out = {}, {}
    handles = []

    for li, layer in enumerate(layers):
        def capture_mlp(_module, args, output, index=li):
            current_mlp[index] = args[0]
            current_mlp_out[index] = output
            output.retain_grad()

        def capture_attn(_module, args, output, index=li):
            current_attn[index] = args[0]
            current_attn_out[index] = output
            output.retain_grad()

        handles.append(layer.fc2.register_forward_hook(capture_mlp))
        handles.append(layer.self_attn.out_proj.register_forward_hook(capture_attn))

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embed_handle = model.model.decoder.embed_tokens.register_forward_hook(
        lambda _module, _args, output: output.detach().requires_grad_(True))
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed + 41)
    total_passes = len(sequences) * n_mc
    quota = max(1, math.ceil(max_tokens / max(1, total_passes)))
    try:
        with torch.enable_grad():
            for sequence in sequences:
                for _ in range(n_mc):
                    current_mlp.clear(); current_attn.clear()
                    current_mlp_out.clear(); current_attn_out.clear()
                    logits = model(sequence.to(next(model.parameters()).device),
                                   use_cache=False).logits[:, :-1].float()
                    probabilities = torch.softmax(logits.detach(), -1)
                    labels = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]),
                                               1, generator=generator).reshape(logits.shape[:-1])
                    score = torch.log_softmax(logits, -1).gather(
                        -1, labels.unsqueeze(-1)).mean()
                    score.backward()
                    for li in range(len(layers)):
                        mlp_value = current_mlp[li].reshape(-1, current_mlp[li].shape[-1])
                        mlp_score = current_mlp_out[li].grad.reshape(
                            -1, current_mlp_out[li].grad.shape[-1])
                        attn_value = current_attn[li].reshape(-1, current_attn[li].shape[-1])
                        attn_score = current_attn_out[li].grad.reshape(
                            -1, current_attn_out[li].grad.shape[-1])
                        count = min(quota, mlp_value.shape[0])
                        positions = torch.linspace(
                            0, mlp_value.shape[0] - 1, count,
                            device=mlp_value.device).long()
                        mlp[li].append(mlp_value[positions].detach().float().cpu())
                        mlp_grad[li].append(mlp_score[positions].detach().float().cpu())
                        attn[li].append(attn_value[positions].detach().float().cpu())
                        attn_grad[li].append(attn_score[positions].detach().float().cpu())
    finally:
        embed_handle.remove()
        for handle in handles:
            handle.remove()
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
    return [FisherStats(
        torch.cat(mlp[i], 0)[:max_tokens],
        torch.cat(mlp_grad[i], 0)[:max_tokens],
        torch.cat(attn[i], 0)[:max_tokens],
        torch.cat(attn_grad[i], 0)[:max_tokens])
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
    kinds = set(kinds)
    selected = []
    for kind in ("unit_fisher", "unit_fisher_pair", "head_fisher", "head_fisher_pair",
                 "unit_fisher_quantized", "head_fisher_quantized",
                 "unit_mean", "unit_off", "unit_merge", "head_mean", "head_off",
                 "attn_uniform", "qk_rank", "ov_rank", "weight_zero"):
        if kind in kinds:
            selected.extend(_top_family(candidates, kind, count))
    return selected


@torch.inference_mode()
def propose(model, stats, args, kinds=ALL_MOVES, fisher_stats=None, layer_indices=None):
    kinds = set(kinds)
    classic_units = bool(kinds & {"unit_mean", "unit_off", "unit_merge"})
    classic_heads = bool(kinds & {"head_mean", "head_off", "attn_uniform",
                                  "qk_rank", "ov_rank"})
    all_candidates = []
    d = model.config.hidden_size
    H = model.config.num_attention_heads
    width = d // H
    indices = (range(len(model.model.decoder.layers))
               if layer_indices is None else layer_indices)
    for li in indices:
        layer = model.model.decoder.layers[li]
        stat = stats[li]
        acts, attended = stat.mlp_sample, stat.attn_sample
        alive_u = live_units(layer)
        unit_ids = alive_u.nonzero().flatten()
        if unit_ids.numel():
            if classic_units:
                A = acts[:, unit_ids].float()
                cols = layer.fc2.weight[:, unit_ids].float().cpu()
                mean = stat.mlp_mean[unit_ids]
                second = stat.mlp_second[unit_ids]
                variance = (second - mean.square()).clamp_min(0)
                colnorm = cols.square().sum(0)
                mean_score = variance * colnorm
                off_score = second * colnorm
                size = min(args.unit_chunk, unit_ids.numel())
                for kind, score in (("unit_mean", mean_score), ("unit_off", off_score)):
                    chosen = torch.argsort(score)[:size]
                    ids = unit_ids[chosen]
                    savings = affected_nonzero(layer.fc1.weight[ids], layer.fc1.bias[ids],
                                               layer.fc2.weight[:, ids])
                    all_candidates.append(Candidate(kind, li, float(score[chosen].sum()), savings,
                                                    {"ids": ids, "means": mean[chosen],
                                                     "label": f"n{len(ids)}"}))

                pool_n = min(args.merge_pool, unit_ids.numel())
                pool_pos = torch.argsort(mean_score)[:pool_n]
                pool_ids, X = unit_ids[pool_pos], A[:, pool_pos]
                Xc = X - X.mean(0)
                sd = Xc.square().mean(0).sqrt()
                Z = Xc / sd.clamp_min(1e-12)
                corr = Z.T @ Z / max(1, len(Z))
                corr.fill_diagonal_(0)
                pairs, used = [], set()
                flat = torch.argsort(corr.abs().flatten(), descending=True)
                for value in flat.tolist():
                    source, target = divmod(value, pool_n)
                    if source == target or source in used or target in used:
                        continue
                    if float(sd[target]) < 1e-12:
                        continue
                    alpha = float((Xc[:, source] * Xc[:, target]).mean() /
                                  Xc[:, target].square().mean().clamp_min(1e-12))
                    beta = float(X[:, source].mean() - alpha * X[:, target].mean())
                    residual = X[:, source] - alpha * X[:, target] - beta
                    proxy = float(residual.square().mean() * colnorm[pool_pos[source]])
                    pairs.append((int(pool_ids[source]), int(pool_ids[target]), alpha, beta, proxy))
                    used.update((source, target))
                    if len(pairs) >= args.merge_chunk:
                        break
                if pairs:
                    sources = torch.tensor([p[0] for p in pairs])
                    savings = affected_nonzero(layer.fc1.weight[sources], layer.fc1.bias[sources],
                                               layer.fc2.weight[:, sources])
                    all_candidates.append(Candidate("unit_merge", li, sum(p[4] for p in pairs), savings,
                                                    {"pairs": pairs, "label": f"n{len(pairs)}"}))

            if kinds & {"unit_fisher", "unit_fisher_pair", "unit_fisher_quantized"}:
                fstat = fisher_stats[li]
                live_acts = fstat.mlp_sample[:, unit_ids]
                gradients = fstat.mlp_grad
                outgoing = layer.fc2.weight[:, unit_ids].detach().cpu()
                directional = gradients @ outgoing
                rough = (live_acts * directional).square().mean(0)
                if kinds & {"unit_fisher", "unit_fisher_quantized"}:
                    if unit_ids.numel() > 1:
                        remove = torch.argmin(rough).reshape(1)
                        fit = grouped_fisher_compensation(
                            live_acts, torch.ones(len(live_acts)), outgoing, remove,
                            downstream_jacobian=gradients)
                        ids = unit_ids[remove]
                        savings = affected_nonzero(
                            layer.fc1.weight[ids], layer.fc1.bias[ids],
                            layer.fc2.weight[:, ids])
                        if "unit_fisher" in kinds:
                            all_candidates.append(Candidate(
                                "unit_fisher", li, fit.predicted_cost, savings,
                                {"ids": ids,
                                 "survivor_ids": unit_ids[fit.survivor_indices],
                                 "delta_out": fit.delta_weights,
                                 "delta_bias": fit.delta_bias,
                                 "label": f"n{len(ids)}"}))
                        if "unit_fisher_quantized" in kinds:
                            quantized = quantize_grouped_fisher_compensation(
                                fit, live_acts, torch.ones(len(live_acts)), outgoing,
                                remove, args.fisher_quant_gamma,
                                downstream_jacobian=gradients)
                            all_candidates.append(Candidate(
                                "unit_fisher_quantized", li,
                                quantized.predicted_cost, savings,
                                {"ids": ids,
                                 "survivor_ids": unit_ids[quantized.survivor_indices],
                                 "delta_out": quantized.delta_weights,
                                 "delta_bias": quantized.delta_bias,
                                 "label": f"n{len(ids)}",
                                 **quantization_metadata(quantized)}))
                if "unit_fisher_pair" in kinds and unit_ids.numel() >= 2:
                    pool = torch.argsort(rough)[:args.fisher_unit_pair_pool]
                    pair_fits = grouped_fisher_pair_compensations(
                        live_acts, torch.ones(len(live_acts)), outgoing,
                        downstream_jacobian=gradients,
                        candidate_indices=pool, limit=args.per_family)
                    for pair, fit in pair_fits:
                        ids = unit_ids[pair]
                        savings = affected_nonzero(
                            layer.fc1.weight[ids], layer.fc1.bias[ids],
                            layer.fc2.weight[:, ids])
                        all_candidates.append(Candidate(
                            "unit_fisher_pair", li, fit.predicted_cost, savings,
                            {"ids": ids,
                             "survivor_ids": unit_ids[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "label": f"n{int(ids[0])}+{int(ids[1])}"}))

        alive_h = live_heads(layer, H)
        fisher_head_ids = set()
        if (kinds & {"head_fisher", "head_fisher_pair", "head_fisher_quantized"}
                and bool(alive_h.any())):
            fstat = fisher_stats[li]
            head_scores = []
            for head in alive_h.nonzero().flatten().tolist():
                sl = slice(head * width, (head + 1) * width)
                contribution = (fstat.attn_sample[:, sl]
                                @ layer.self_attn.out_proj.weight[:, sl].float().cpu().T)
                head_scores.append((float((fstat.attn_grad * contribution).sum(1)
                                          .square().mean()), head))
            fisher_head_ids = {head for _, head in sorted(head_scores)
                               [:args.fisher_heads_per_layer]}
        for head in alive_h.nonzero().flatten().tolist():
            sl = slice(head * width, (head + 1) * width)
            common = (layer.self_attn.q_proj.weight[sl], layer.self_attn.q_proj.bias[sl],
                      layer.self_attn.k_proj.weight[sl], layer.self_attn.k_proj.bias[sl],
                      layer.self_attn.v_proj.weight[sl], layer.self_attn.v_proj.bias[sl],
                      layer.self_attn.out_proj.weight[:, sl])
            savings = affected_nonzero(*common)
            if classic_heads:
                contribution = (attended[:, sl].float()
                                @ layer.self_attn.out_proj.weight[:, sl].float().cpu().T)
                centered = contribution - contribution.mean(0)
                all_candidates.append(Candidate(
                    "head_mean", li, float(centered.square().mean()), savings,
                    {"head": head, "mean": contribution.mean(0), "label": f"h{head}"}))
                all_candidates.append(Candidate(
                    "head_off", li, float(contribution.square().mean()), savings,
                    {"head": head, "label": f"h{head}"}))

            if head in fisher_head_ids:
                fstat = fisher_stats[li]
                live_dims = torch.cat([
                    torch.arange(h * width, (h + 1) * width)
                    for h in alive_h.nonzero().flatten().tolist()])
                remove = ((live_dims // width) == head).nonzero().flatten()
                fit = grouped_fisher_compensation(
                    fstat.attn_sample[:, live_dims],
                    torch.ones(len(fstat.attn_sample)),
                    layer.self_attn.out_proj.weight[:, live_dims].detach().cpu(),
                    remove, downstream_jacobian=fstat.attn_grad)
                if "head_fisher" in kinds:
                    all_candidates.append(Candidate(
                        "head_fisher", li, fit.predicted_cost, savings,
                        {"head": head,
                         "survivor_dims": live_dims[fit.survivor_indices],
                         "delta_out": fit.delta_weights,
                         "delta_bias": fit.delta_bias,
                         "label": f"h{head}"}))
                if "head_fisher_quantized" in kinds:
                    quantized = quantize_grouped_fisher_compensation(
                        fit, fstat.attn_sample[:, live_dims],
                        torch.ones(len(fstat.attn_sample)),
                        layer.self_attn.out_proj.weight[:, live_dims].detach().cpu(),
                        remove, args.fisher_quant_gamma,
                        downstream_jacobian=fstat.attn_grad,
                        group_ids=live_dims // width)
                    all_candidates.append(Candidate(
                        "head_fisher_quantized", li,
                        quantized.predicted_cost, savings,
                        {"head": head,
                         "survivor_dims": live_dims[quantized.survivor_indices],
                         "delta_out": quantized.delta_weights,
                         "delta_bias": quantized.delta_bias,
                         "label": f"h{head}",
                         **quantization_metadata(quantized)}))

                if ("head_fisher_pair" in kinds and head == min(fisher_head_ids)
                        and len(fisher_head_ids) >= 2):
                    for pair in torch.combinations(torch.tensor(sorted(fisher_head_ids)), 2):
                        pair_heads = pair.tolist()
                        pair_remove = torch.cat([
                            ((live_dims // width) == h).nonzero().flatten()
                            for h in pair_heads])
                        pair_fit = grouped_fisher_compensation(
                            fstat.attn_sample[:, live_dims],
                            torch.ones(len(fstat.attn_sample)),
                            layer.self_attn.out_proj.weight[:, live_dims].detach().cpu(),
                            pair_remove, downstream_jacobian=fstat.attn_grad)
                        dims = torch.cat([
                            torch.arange(h * width, (h + 1) * width)
                            for h in pair_heads])
                        pair_savings = affected_nonzero(
                            layer.self_attn.q_proj.weight[dims],
                            layer.self_attn.q_proj.bias[dims],
                            layer.self_attn.k_proj.weight[dims],
                            layer.self_attn.k_proj.bias[dims],
                            layer.self_attn.v_proj.weight[dims],
                            layer.self_attn.v_proj.bias[dims],
                            layer.self_attn.out_proj.weight[:, dims])
                        all_candidates.append(Candidate(
                            "head_fisher_pair", li, pair_fit.predicted_cost,
                            pair_savings,
                            {"heads": pair_heads,
                             "survivor_dims": live_dims[pair_fit.survivor_indices],
                             "delta_out": pair_fit.delta_weights,
                             "delta_bias": pair_fit.delta_bias,
                             "label": f"h{pair_heads[0]}+h{pair_heads[1]}"}))

            qk_tensors = (layer.self_attn.q_proj.weight[sl], layer.self_attn.q_proj.bias[sl],
                          layer.self_attn.k_proj.weight[sl], layer.self_attn.k_proj.bias[sl])
            if "attn_uniform" in kinds:
                qk_savings = affected_nonzero(*qk_tensors)
                qk_proxy = sum(float(t.float().square().sum()) for t in qk_tensors)
                all_candidates.append(Candidate("attn_uniform", li, qk_proxy, qk_savings,
                                                {"head": head, "label": f"h{head}"}))

            active_qk = ((layer.self_attn.q_proj.weight[sl].abs().sum(1) > 0) &
                         (layer.self_attn.k_proj.weight[sl].abs().sum(1) > 0)).nonzero().flatten()
            if "qk_rank" in kinds and active_qk.numel() > args.rank_chunk:
                A = torch.cat([layer.self_attn.q_proj.weight[sl][active_qk].T,
                               layer.self_attn.q_proj.bias[sl][active_qk][None]], 0)
                B = layer.self_attn.k_proj.weight[sl][active_qk].T
                singular = thin_singular_values(A, B)
                cut = min(args.rank_chunk, active_qk.numel() - 1)
                proxy = float(singular[-cut:].square().sum())
                rank = active_qk.numel() - cut
                current = affected_nonzero(*qk_tensors)
                savings = current - rank * (2 * d + 1)
                all_candidates.append(Candidate("qk_rank", li, proxy, savings,
                                                {"head": head, "cut": cut,
                                                 "label": f"h{head}r-{cut}"}))

            active_ov = ((layer.self_attn.v_proj.weight[sl].abs().sum(1) > 0) &
                         (layer.self_attn.out_proj.weight[:, sl].abs().sum(0) > 0)).nonzero().flatten()
            if "ov_rank" in kinds and active_ov.numel() > args.rank_chunk:
                A = torch.cat([layer.self_attn.v_proj.weight[sl][active_ov].T,
                               layer.self_attn.v_proj.bias[sl][active_ov][None]], 0)
                B = layer.self_attn.out_proj.weight[:, sl][:, active_ov]
                singular = thin_singular_values(A, B)
                cut = min(args.rank_chunk, active_ov.numel() - 1)
                proxy = float(singular[-cut:].square().sum())
                savings = cut * (2 * d + 1)
                all_candidates.append(Candidate("ov_rank", li, proxy, savings,
                                                {"head": head, "cut": cut,
                                                 "label": f"h{head}r-{cut}"}))

        if "weight_zero" in kinds:
            for name, module in linear_layers(layer).items():
                W = module.weight
                live = (W != 0).flatten().nonzero().flatten()
                if not live.numel():
                    continue
                size = min(args.weight_chunk, live.numel())
                values = W.flatten()[live].abs()
                pos = torch.topk(values, size, largest=False).indices
                chosen = live[pos]
                proxy = float(W.flatten()[chosen].float().square().sum())
                all_candidates.append(Candidate("weight_zero", li, proxy, size,
                                                {"module": name, "indices": chosen,
                                                 "label": f"{name}.n{size}"}))

    return select_proposals(all_candidates, kinds, args.per_family)


@torch.inference_mode()
def _apply_candidate(model, candidate: Candidate):
    layer = model.model.decoder.layers[candidate.layer]
    p = candidate.payload
    if candidate.kind in ("unit_fisher", "unit_fisher_pair", "unit_fisher_quantized"):
        ids = p["ids"]
        survivors = p["survivor_ids"]
        layer.fc2.weight[:, survivors] = (layer.fc2.weight[:, survivors]
            + p["delta_out"].to(layer.fc2.weight))
        layer.fc2.bias.add_(p["delta_bias"].to(layer.fc2.bias))
        layer.fc1.weight[ids] = 0
        layer.fc1.bias[ids] = 0
        layer.fc2.weight[:, ids] = 0
        return
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
    if candidate.kind in ("head_fisher", "head_fisher_pair", "head_fisher_quantized"):
        survivors = p["survivor_dims"]
        layer.self_attn.out_proj.weight[:, survivors] = (
            layer.self_attn.out_proj.weight[:, survivors]
            + p["delta_out"].to(layer.self_attn.out_proj.weight))
        layer.self_attn.out_proj.bias.add_(
            p["delta_bias"].to(layer.self_attn.out_proj.bias))
        width = layer.self_attn.head_dim
        heads = (p["heads"] if candidate.kind == "head_fisher_pair"
                 else [p["head"]])
        for head in heads:
            sl = slice(head * width, (head + 1) * width)
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                               layer.self_attn.v_proj):
                projection.weight[sl].zero_(); projection.bias[sl].zero_()
            layer.self_attn.out_proj.weight[:, sl].zero_()
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
    p.add_argument("--fisher-tokens", type=int, default=128)
    p.add_argument("--fisher-sequences", type=int, default=2)
    p.add_argument("--fisher-mc", type=int, default=1)
    p.add_argument("--fisher-heads-per-layer", type=int, default=2)
    p.add_argument("--fisher-unit-pair-pool", type=int, default=8)
    p.add_argument("--move-group", choices=("classic", "classic_weights",
                                             "fisher_single", "fisher",
                                             "fisher_quantized", "all"),
                   default="classic")
    p.add_argument("--fisher-quant-gamma", type=float, default=.2)
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-after-stop", action="store_true")
    args = p.parse_args()
    if args.fisher_quant_gamma <= 0:
        p.error("--fisher-quant-gamma must be positive")
    if args.rho < 0 or args.n_sigma < 0:
        p.error("--rho and --n-sigma must be non-negative")
    if min(args.nsamples, args.seqlen, args.val_a, args.val_b,
           args.stats_tokens, args.fisher_tokens, args.fisher_sequences,
           args.fisher_mc, args.refresh_every, args.per_family,
           args.calibration_batch_size) < 1:
        p.error("sample, sequence, Fisher, refresh, and candidate counts must be positive")
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
        "stats_tokens": args.stats_tokens, "fisher_tokens": args.fisher_tokens,
        "stats_sequences": args.stats_sequences,
        "calibration_batch_size": args.calibration_batch_size,
        "fisher_sequences": args.fisher_sequences, "fisher_mc": args.fisher_mc,
        "fisher_quant_gamma": args.fisher_quant_gamma,
        "refresh_every": args.refresh_every, "unit_chunk": args.unit_chunk,
        "merge_pool": args.merge_pool, "merge_chunk": args.merge_chunk,
        "rank_chunk": args.rank_chunk, "weight_chunk": args.weight_chunk,
        "per_family": args.per_family,
        "fisher_heads_per_layer": args.fisher_heads_per_layer,
        "fisher_unit_pair_pool": args.fisher_unit_pair_pool,
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
    kinds = ({"classic": CLASSIC_MOVES, "classic_weights": ALL_MOVES,
              "fisher_single": FISHER_SINGLE_MOVES,
              "fisher": FISHER_MOVES, "fisher_quantized": FISHER_QUANTIZED_MOVES,
              "all": ALL_MOVES + FISHER_MOVES}[args.move_group])
    stats, fstats, proposal_cache = None, None, None
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
            fstats = (fisher_samples(
                model, calibration[:args.fisher_sequences], args.fisher_tokens,
                args.fisher_mc, args.seed + step)
                if any(kind in kinds for kind in FISHER_ALL_MOVES) else None)
            proposal_cache = None
            dirty_layer = None
        if fstats is None:
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
        else:
            candidates = propose(model, stats, args, kinds, fstats)
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
               "quant_case": chosen.payload.get("quant_case"),
               "quant_weight_groups": chosen.payload.get("quant_weight_groups"),
               "quant_total_groups": chosen.payload.get("quant_total_groups"),
               "quant_bias": chosen.payload.get("quant_bias"),
               "quant_step": chosen.payload.get("quant_step"),
               "matched_baselines": sorted(completed_targets),
               "seconds": time.perf_counter() - started}
        rows.append(row)
        write_checkpoint(args, rows, "running", signature)
        print(f"step {step}: {chosen.label}, removed {100 * removed / prunable_total:.3f}%, "
              f"valB dL={delta:.4g}, {row['seconds']:.0f}s", flush=True)
        if any(kind in kinds for kind in FISHER_ALL_MOVES):
            stats, fstats, proposal_cache = None, None, None
        elif step % args.refresh_every == 0:
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

    if args.published_opt:
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
                           "classic uses an additional C4 selection split"
                           if args.published_opt else "CPU-close OPT pilot"),
              "moves": list(kinds),
              "fisher_quant_gamma": args.fisher_quant_gamma,
              "calibration": {"dataset": "c4", "nsamples": args.nsamples,
                              "seqlen": args.seqlen, "stats_tokens": args.stats_tokens,
                              "stats_sequences": (args.stats_sequences or args.nsamples),
                              "fisher_tokens": args.fisher_tokens,
                              "fisher_sequences": args.fisher_sequences,
                              "fisher_mc": args.fisher_mc},
              "selection": {"dataset": args.selection_dataset,
                            "seed": args.selection_seed, "val_a": args.val_a,
                            "val_b": args.val_b, "rho": args.rho,
                            "n_sigma": args.n_sigma},
              "candidate_budget": {"unit_chunk": args.unit_chunk,
                                   "fisher_unit_shortlist_per_layer": 1,
                                   "fisher_heads_per_layer": args.fisher_heads_per_layer,
                                   "fisher_unit_pair_pool": args.fisher_unit_pair_pool,
                                   "merge_pool": args.merge_pool,
                                   "merge_chunk": args.merge_chunk,
                                   "rank_chunk": args.rank_chunk,
                                   "per_family": args.per_family},
              "fisher_screening": ("one rough-shortlisted MLP unit per layer and "
                                   "a bounded head shortlist"
                                   if any(kind in kinds for kind in FISHER_ALL_MOVES)
                                   else None),
              "loss_targets": targets,
              "termination_reason": termination_reason,
              "stop_index": stop_index,
              "stopping_point": rows[stop_index] if stop_index is not None else None,
              "accepted_endpoint": endpoint,
              "rejected_candidate": rejected_candidate,
              "quantization_counts": {
                  case: sum(row.get("quant_case") == case for row in rows)
                  for case in ("off", "bias_only", "weights_only", "weights_and_bias")
                  if any(row.get("quant_case") == case for row in rows)},
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    if not args.no_model_save:
        tmp = args.model_checkpoint.with_suffix(args.model_checkpoint.suffix + ".tmp")
        torch.save({"model": model.state_dict(), "rows": rows,
                    "signature": signature}, tmp)
        os.replace(tmp, args.model_checkpoint)
    write_checkpoint(args, rows, "complete", signature)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
