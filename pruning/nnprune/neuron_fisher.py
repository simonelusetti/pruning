from __future__ import annotations

import torch


def _check(incoming, outgoing, a_in, b_in, a_out, b_out, deleted):
    hidden, width = incoming.shape
    if (outgoing.shape[1] != hidden + 1 or
            a_in.shape != (width, width) or b_in.shape != (hidden, hidden) or
            a_out.shape != (hidden + 1, hidden + 1) or
            b_out.shape != (outgoing.shape[0],) * 2 or
            deleted.shape != (hidden,) or deleted.dtype != torch.bool):
        raise ValueError("hidden-neuron factors and augmented weights do not match")


@torch.no_grad()
def neuron_obs_scores(incoming, outgoing, a_in, b_in, a_out, b_out, deleted):
    """Singleton compensated costs for incoming and outgoing cuts of one neuron."""
    _check(incoming, outgoing, a_in, b_in, a_out, b_out, deleted)
    incoming = incoming.double()
    outgoing = outgoing.to(incoming)
    a_in, b_in, a_out, b_out = (value.to(incoming)
                               for value in (a_in, b_in, a_out, b_out))
    active = torch.nonzero(~deleted.to(incoming.device)).flatten()
    scores = incoming.new_full((len(incoming), 2), float("inf"))
    if not len(active):
        return scores
    inverse_b = torch.cholesky_inverse(
        torch.linalg.cholesky(b_in[active][:, active])).diag()
    columns = torch.cat((active, active.new_tensor([len(incoming)])))
    inverse_a = torch.cholesky_inverse(
        torch.linalg.cholesky(a_out[columns][:, columns])).diag()[:-1]
    rows = incoming[active]
    weights = outgoing[:, active]
    scores[active, 0] = .5 * (rows * (rows @ a_in)).sum(1) / inverse_b
    scores[active, 1] = .5 * (weights * (b_out @ weights)).sum(0) / inverse_a
    return scores


@torch.no_grad()
def select_neuron_cuts(scores, savings, required):
    """Select distinct neurons globally by singleton cost per canonical saving."""
    if required < 0 or len(scores) != len(savings):
        raise ValueError("invalid neuron removal budget")
    costs, direction = torch.cat(scores).min(1)
    counts = torch.cat(savings)
    eligible = torch.nonzero(torch.isfinite(costs) & (counts > 0)).flatten()
    if required > int(counts[eligible].sum()):
        raise ValueError("target exceeds remaining hidden-neuron capacity")
    chosen = torch.zeros_like(costs, dtype=torch.bool)
    if required:
        order = eligible[torch.argsort(costs[eligible] / counts[eligible], stable=True)]
        cumulative = counts[order].cumsum(0)
        count = int(torch.searchsorted(cumulative, cumulative.new_tensor(required))) + 1
        chosen[order[:count]] = True
    incoming = chosen & (direction == 0)
    outgoing = chosen & (direction == 1)
    sizes = [len(value) for value in scores]
    return list(zip(incoming.split(sizes), outgoing.split(sizes)))


@torch.no_grad()
def neuron_complement(incoming, outgoing, a_in, b_in, a_out, b_out,
                      deleted, incoming_cut, outgoing_cut):
    """Optimize the primary move; future canonicalization coordinates stay frozen."""
    _check(incoming, outgoing, a_in, b_in, a_out, b_out, deleted)
    incoming = incoming.double()
    outgoing = outgoing.to(incoming)
    a_in, b_in, a_out, b_out = (value.to(incoming)
                               for value in (a_in, b_in, a_out, b_out))
    deleted, incoming_cut, outgoing_cut = (
        value.to(device=incoming.device) for value in
        (deleted, incoming_cut, outgoing_cut))
    if (incoming_cut.shape != deleted.shape or outgoing_cut.shape != deleted.shape or
            incoming_cut.dtype != torch.bool or outgoing_cut.dtype != torch.bool or
            bool((incoming_cut & outgoing_cut).any()) or
            bool((deleted & (incoming_cut | outgoing_cut)).any())):
        raise ValueError("neuron cut sets must be disjoint and currently active")
    removed = deleted | incoming_cut | outgoing_cut
    surviving = torch.nonzero(~removed).flatten()
    input_ids = torch.nonzero(incoming_cut).flatten()
    output_ids = torch.nonzero(outgoing_cut).flatten()
    free_columns = torch.cat((surviving, surviving.new_tensor([len(incoming)])))
    delta_in, delta_out = torch.zeros_like(incoming), torch.zeros_like(outgoing)
    delta_in[input_ids] = -incoming[input_ids]
    delta_out[:, output_ids] = -outgoing[:, output_ids]
    cut_rows, cut_columns = incoming[input_ids], outgoing[:, output_ids]
    fixed_in = b_in[input_ids][:, input_ids] @ cut_rows
    fixed_out = cut_columns @ a_out[output_ids][:, output_ids]
    no_update = (.5 * (cut_rows * (fixed_in @ a_in)).sum() +
                 .5 * (cut_columns * (b_out @ fixed_out)).sum())
    residuals = []
    if len(input_ids) and len(surviving):
        rhs = b_in[surviving][:, input_ids] @ incoming[input_ids]
        factor = b_in[surviving][:, surviving]
        update = torch.cholesky_solve(rhs, torch.linalg.cholesky(factor))
        delta_in[surviving] = update
        residuals.append(float((factor @ update - rhs).norm() / rhs.norm().clamp_min(1e-30)))
    if len(output_ids):
        rhs = outgoing[:, output_ids] @ a_out[output_ids][:, free_columns]
        factor = a_out[free_columns][:, free_columns]
        update = torch.cholesky_solve(rhs.T, torch.linalg.cholesky(factor)).T
        delta_out[:, free_columns] = update
        residuals.append(float((update @ factor - rhs).norm() / rhs.norm().clamp_min(1e-30)))
    # Free-coordinate gradients vanish at the optimum; only cut coordinates contribute.
    fixed_in -= b_in[input_ids][:, surviving] @ delta_in[surviving]
    fixed_out -= delta_out[:, free_columns] @ a_out[free_columns][:, output_ids]
    damage = (.5 * (cut_rows * (fixed_in @ a_in)).sum() +
              .5 * (cut_columns * (b_out @ fixed_out)).sum())
    if float(damage - no_update) > 1e-8 * max(1., abs(float(no_update))):
        raise RuntimeError("neuron compensation increased its quadratic objective")
    revised_in, revised_out = incoming + delta_in, outgoing + delta_out
    revised_in[input_ids] = 0
    revised_out[:, output_ids] = 0
    updates_in = delta_in[surviving]
    updates_out = delta_out[:, free_columns]
    return revised_in, revised_out, {
        "incoming_cut_neurons": input_ids.tolist(),
        "outgoing_cut_neurons": output_ids.tolist(),
        "predicted_fisher_damage": float(damage),
        "no_update_fisher_damage": float(no_update),
        "relative_residual": max(residuals, default=0.),
        "updates": int(torch.count_nonzero(updates_in) + torch.count_nonzero(updates_out)),
        "update_norm": float((updates_in.square().sum() + updates_out.square().sum()).sqrt()),
    }
