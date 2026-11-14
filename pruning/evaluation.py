from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def paired(values, reference):
    difference = values.double() - reference.double()
    return float(difference.mean()), float(difference.std(unbiased=True) / math.sqrt(len(difference)))


def loss_change(values, reference):
    mean, se = paired(values, reference)
    return {"loss": float(values.mean()), "delta": mean, "se": se, "variance": float(values.var()),
            "damage_percent": 100 * (float(values.mean() / reference.mean()) - 1)}


@torch.inference_mode()
def language_losses(model, sequences, batch_size=2):
    device = next(model.parameters()).device
    batches = DataLoader(TensorDataset(torch.cat(sequences)), batch_size=batch_size)
    losses = []
    for (ids,) in batches:
        ids = ids.to(device)
        logits = model(ids, use_cache=False).logits[:, :-1].float()
        values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 ids[:, 1:].reshape(-1), reduction="none")
        losses.append(values.reshape(len(ids), -1).mean(1).cpu())
    return torch.cat(losses).double()


def choose_candidate(problem, proposals, protocol, *, filter_selection=False):
    before = problem.snapshot()
    count = problem.remaining()
    evaluated = []
    reference = problem.selection_reference() if protocol.candidate_selection == "loss" else None
    for proposal in proposals:
        problem.restore(before)
        if proposal["apply"]() is False:
            continue
        saved = count - problem.remaining()
        if saved <= 0:
            continue
        if reference is None:
            cost = proposal.get("fisher_cost_per_removed",
                                proposal.get("fisher_cost", proposal.get("proxy_cost", float("inf"))))
            evaluated.append((proposal, saved, cost, None))
        else:
            values = problem.losses("selection")
            mean, se = paired(values, reference)
            if filter_selection and mean > protocol.n_sigma * se + protocol.rho * float(reference.mean()):
                continue
            evaluated.append((proposal, saved, float(values.mean()), values))
    problem.restore(before)
    if not evaluated:
        return None
    best = min(evaluated, key=lambda item: item[2])
    ties = ([item for item in evaluated if paired(item[3], best[3])[0] <=
             protocol.n_sigma * paired(item[3], best[3])[1]] if reference is not None else [best])
    chosen = min(ties, key=lambda item: (-item[1], item[2]))
    chosen[0]["apply"]()
    return {key: value for key, value in chosen[0].items() if key != "apply"} | {
        "actual_savings": chosen[1], "n_candidates": len(evaluated), "n_ties": len(ties)}
