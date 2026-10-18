"""
Sigmoid MLP on two-moons (binary, Bernoulli likelihood) -- the pruningNN setting.

Boundary moves for a hidden sigmoid unit i of layer l
-----------------------------------------------------
* unit_sat  c=0 / c=1    b_i -> -inf / +inf.  For l >= 1 the inputs are sigmoid
                          outputs, strictly positive, so ANY incoming weight
                          w_ij -> -inf / +inf reaches the same endpoint: the
                          endpoint is unit-level even when the path is a single
                          weight.  Realised exactly by folding c * W_{l+1}[:, i]
                          into b_{l+1}; no large numbers are stored.
* unit_mean c=mean       whole fan-in -> 0 with b_i = logit(c): the unit is the
                          constant c, chosen as its training mean.  Same fold.
* unit_step (layer 0)     w_ij -> +-inf with sign-varying input x_j: the unit
                          becomes the Heaviside H(s x_j) and its other fan-in and
                          bias become unidentifiable (they are removed).
* unit_merge              a_i ~= alpha a_i2 + beta (OLS on training activations):
                          fold into unit i2's outgoing column and the next bias,
                          then remove i.  The duplicate-neuron singularity.
* unit_ablate             set one unit output to zero and remove its fan-in,
                          bias and fan-out.  Candidates are shortlisted by a
                          virtual-gate Fisher score (or an activation-energy
                          control), then the ordinary structural screen measures
                          the realised deletion's loss exactly.
* unit_fisher             any hidden layer: delete one unit while refitting all
                          surviving outgoing columns and the next bias in the
                          downstream Fisher metric.
Weight moves ('zero') are generic (core.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .core import ACTIVE, REMOVED, STEP, Adapter, Move, State, zero_entries

CLASSIC_MOVES = ("unit_sat", "unit_mean", "unit_step", "unit_merge")
OUTDATED_MOVES = ("unit_ablate",)
FISHER_MOVES = ("unit_fisher", "unit_fisher_pair")
FISHER_QUANTIZED_MOVES = ("unit_fisher_quantized",)
FISHER_QUANTIZED_POST_MOVES = ("unit_fisher_quantized_post",)
EXTENSION_MOVES = FISHER_QUANTIZED_MOVES + FISHER_QUANTIZED_POST_MOVES
MOVE_GROUPS = {
    "classic": CLASSIC_MOVES,
    "outdated": OUTDATED_MOVES,
    "extension": EXTENSION_MOVES,
    "fisher": FISHER_MOVES,
    "fisher_general": ("fisher_general",),
}


@dataclass(frozen=True)
class GroupedFisherCompensation:
    """Optimal outgoing-side compensation for deleting activation columns."""

    survivor_indices: torch.Tensor
    delta_weights: torch.Tensor
    delta_bias: torch.Tensor
    predicted_cost: float
    rank: int
    rcond: float
    quantization_step: float | None = None
    quantized_weight_groups: int | None = None
    total_weight_groups: int | None = None
    quantized_bias: bool | None = None
    quantization_case: str | None = None


def _svd_lstsq(A, b, rcond):
    if A.shape[1] > A.shape[0]:
        gram = A @ A.T
        values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
        top = values[-1].clamp_min(0) if values.numel() else values.new_zeros(())
        keep = values > top * rcond ** 2
        if bool(keep.any()):
            dual = vectors[:, keep] @ ((vectors[:, keep].T @ b) / values[keep, None])
            return A.T @ dual, int(keep.sum())
        return torch.zeros(A.shape[1], b.shape[1], dtype=A.dtype, device=A.device), 0
    U, singular, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = singular > singular[0] * rcond if singular.numel() else singular.bool()
    if bool(keep.any()):
        solution = Vh[keep].T @ ((U[:, keep].T @ b) / singular[keep, None])
    else:
        solution = torch.zeros(A.shape[1], b.shape[1], dtype=A.dtype, device=A.device)
    return solution, int(keep.sum())


def grouped_fisher_compensation(activations: torch.Tensor,
                                fisher_weights: torch.Tensor,
                                outgoing_weights: torch.Tensor,
                                remove_index,
                                downstream_jacobian: torch.Tensor | None = None,
                                adjustable_mask: torch.Tensor | None = None,
                                rcond=None) -> GroupedFisherCompensation:
    """Solve the grouped-Fisher neuron deletion problem in prediction space.

    ``activations`` has one column per live unit.  The final column of the
    feature design is a constant, so the returned update includes the next
    layer's bias.  ``outgoing_weights`` may be a vector for the final hidden
    layer or a ``[next_units, units]`` matrix.  In the latter case,
    ``downstream_jacobian[n, j]`` is the derivative of example n's final logit
    with respect to next-layer preactivation j.  The resulting regression
    design is ``J(x) tensor [a_R(x), 1]``.

    A compact tensor-product basis is used when every compensation parameter
    is adjustable.  It is algebraically the same least-squares problem after
    discarding directions below the source-precision cutoff.  Irregular masks
    use the explicit design.  Both routes use float64 SVD least squares and
    evaluate cost from the weighted prediction residual, avoiding a
    cancellation-prone Schur-complement subtraction.
    """
    if activations.ndim != 2:
        raise ValueError("activations must be a [samples, units] matrix")
    n, m = activations.shape
    if n == 0:
        raise ValueError("activations must be nonempty")
    remove = torch.as_tensor(remove_index, device=activations.device,
                             dtype=torch.long).reshape(-1)
    if remove.numel() == 0 or bool((remove < 0).any()) or bool((remove >= m).any()):
        raise ValueError("remove_index must identify columns of a nonempty matrix")
    remove = torch.unique(remove, sorted=True)
    if fisher_weights.shape != (n,):
        raise ValueError("fisher_weights must have one entry per sample")
    scalar_output = outgoing_weights.ndim == 1
    if scalar_output:
        if outgoing_weights.shape != (m,):
            raise ValueError("outgoing_weights must have one entry per unit")
        outgoing_weights = outgoing_weights.unsqueeze(0)
    elif outgoing_weights.ndim != 2 or outgoing_weights.shape[1] != m:
        raise ValueError("outgoing_weights must be [next_units, units]")
    next_units = outgoing_weights.shape[0]
    if downstream_jacobian is None:
        if next_units != 1:
            raise ValueError("downstream_jacobian is required before a hidden next layer")
        downstream_jacobian = torch.ones(n, 1, dtype=activations.dtype,
                                         device=activations.device)
    if downstream_jacobian.shape != (n, next_units):
        raise ValueError("downstream_jacobian must be [samples, next_units]")

    source_dtype = activations.dtype
    A = activations.detach().to(dtype=torch.float64)
    q = fisher_weights.detach().to(dtype=torch.float64).clamp_min(0)
    V = outgoing_weights.detach().to(dtype=torch.float64)
    J = downstream_jacobian.detach().to(dtype=torch.float64)
    keep_mask = torch.ones(m, dtype=torch.bool, device=A.device)
    keep_mask[remove] = False
    keep = keep_mask.nonzero().flatten()
    X = torch.cat((A[:, keep], torch.ones(n, 1, dtype=A.dtype, device=A.device)), dim=1)
    d = X.shape[1]
    if adjustable_mask is None:
        mask = torch.ones(next_units, d, dtype=torch.bool, device=A.device)
    else:
        if adjustable_mask.shape != (next_units, m):
            raise ValueError("adjustable_mask must match outgoing_weights")
        mask = torch.cat((adjustable_mask.detach().to(device=A.device)[:, keep],
                          torch.ones(next_units, 1, dtype=torch.bool, device=A.device)), dim=1)

    # Scaling by sqrt(q / n) makes the squared residual exactly the
    # mean-normalised Fisher quadratic apart from its factor of one half.
    scale = (q / n).sqrt()
    target = (A[:, remove] * (J @ V[:, remove])).sum(1)
    yw = target * scale
    if rcond is None:
        # Conversion to float64 improves the solve but cannot create information
        # below the precision at which activations/Jacobians were measured.
        source_eps = torch.finfo(source_dtype).eps
        rcond = float(source_eps * max(n, next_units * d))

    if bool(mask.all()):
        # J = Jr Vj^T and X = Xr Vx^T.  The row-wise Kronecker design then
        # factors through Z[n,(rj,rx)] = Jr[n,rj] Xr[n,rx].
        Uj, sj, Vhj = torch.linalg.svd(J, full_matrices=False)
        Ux, sx, Vhx = torch.linalg.svd(X, full_matrices=False)
        rj = max(int((sj > sj[0] * rcond).sum()), 1)
        rx = max(int((sx > sx[0] * rcond).sum()), 1)
        Jr, Xr = Uj[:, :rj] * sj[:rj], Ux[:, :rx] * sx[:rx]
        Z = (Jr[:, :, None] * Xr[:, None, :]).reshape(n, rj * rx)
        Zw = Z * scale[:, None]
        solution, rank = _svd_lstsq(Zw, yw[:, None], rcond)
        gamma = solution[:, 0].reshape(rj, rx)
        B = Vhj[:rj].T @ gamma @ Vhx[:rx]
        residual = yw - Zw @ solution[:, 0]
    else:
        design = (J[:, :, None] * X[:, None, :]).reshape(n, next_units * d)
        allowed = mask.reshape(-1).nonzero().flatten()
        Dw = design[:, allowed] * scale[:, None]
        solution, rank = _svd_lstsq(Dw, yw[:, None], rcond)
        beta = torch.zeros(next_units * d, dtype=torch.float64, device=A.device)
        beta[allowed] = solution[:, 0]
        B = beta.reshape(next_units, d)
        residual = yw - Dw @ solution[:, 0]

    cost = 0.5 * residual.square().sum()
    delta_weights = B[:, :-1]
    delta_bias = B[:, -1]
    if scalar_output:
        delta_weights = delta_weights[0]
        delta_bias = delta_bias[0]
    return GroupedFisherCompensation(
        survivor_indices=keep,
        delta_weights=delta_weights,
        delta_bias=delta_bias,
        predicted_cost=float(cost),
        rank=rank,
        rcond=float(rcond),
    )


def grouped_fisher_singleton_compensations(
        activations: torch.Tensor,
        fisher_weights: torch.Tensor,
        outgoing_weights: torch.Tensor,
        downstream_jacobian: torch.Tensor | None = None,
        adjustable_mask: torch.Tensor | None = None,
        rcond=None) -> list[GroupedFisherCompensation]:
    """Solve every single-column deletion from one shared factorisation."""
    if activations.ndim != 2:
        raise ValueError("activations must be a [samples, units] matrix")
    n, m = activations.shape
    if n == 0:
        raise ValueError("activations must be nonempty")
    if fisher_weights.shape != (n,):
        raise ValueError("fisher_weights must have one entry per sample")
    scalar_output = outgoing_weights.ndim == 1
    if scalar_output:
        if outgoing_weights.shape != (m,):
            raise ValueError("outgoing_weights must have one entry per unit")
        outgoing_weights = outgoing_weights.unsqueeze(0)
    elif outgoing_weights.ndim != 2 or outgoing_weights.shape[1] != m:
        raise ValueError("outgoing_weights must be [next_units, units]")
    next_units = outgoing_weights.shape[0]
    if downstream_jacobian is None:
        if next_units != 1:
            raise ValueError("downstream_jacobian is required before a hidden next layer")
        downstream_jacobian = torch.ones(
            n, 1, dtype=activations.dtype, device=activations.device)
    if downstream_jacobian.shape != (n, next_units):
        raise ValueError("downstream_jacobian must be [samples, next_units]")
    if adjustable_mask is not None:
        if adjustable_mask.shape != outgoing_weights.shape:
            raise ValueError("adjustable_mask must match outgoing_weights")
        if not bool(adjustable_mask.all()):
            base_weights = outgoing_weights[0] if scalar_output else outgoing_weights
            return [grouped_fisher_compensation(
                activations, fisher_weights, base_weights, i,
                downstream_jacobian, adjustable_mask, rcond) for i in range(m)]

    source_dtype = activations.dtype
    A = activations.detach().to(dtype=torch.float64)
    q = fisher_weights.detach().to(dtype=torch.float64).clamp_min(0)
    V = outgoing_weights.detach().to(dtype=torch.float64)
    J = downstream_jacobian.detach().to(dtype=torch.float64)
    X = torch.cat((A, torch.ones(n, 1, dtype=A.dtype, device=A.device)), dim=1)
    scale = (q / n).sqrt()
    if rcond is None:
        rcond = float(torch.finfo(source_dtype).eps * max(n, next_units * (m + 1)))

    parameter_count = (m + 1) * next_units
    if parameter_count <= n:
        design = (X[:, :, None] * J[:, None, :]).reshape(n, parameter_count)
        design *= scale[:, None]
        gram = design.T @ design
        factor, info = torch.linalg.cholesky_ex((gram + gram.T) / 2)
        spectrum = torch.linalg.eigvalsh((gram + gram.T) / 2)
        well_conditioned = bool(spectrum[0] > spectrum[-1].clamp_min(0) * rcond ** 2)
        if int(info) == 0 and well_conditioned:
            inverse = torch.cholesky_inverse(factor)
            offsets = torch.arange(next_units, device=A.device)
            indices = (torch.arange(m, device=A.device)[:, None]
                       * next_units + offsets).long()
            blocks = inverse[indices[:, :, None], indices[:, None, :]]
            block_factor, block_info = torch.linalg.cholesky_ex(
                (blocks + blocks.transpose(1, 2)) / 2)
            if bool((block_info == 0).all()):
                rhs = V.T
                multipliers = torch.cholesky_solve(
                    rhs.unsqueeze(-1), block_factor).squeeze(-1)
                costs = 0.5 * (rhs * multipliers).sum(1)
                all_indices = torch.arange(m, device=A.device)
                results = []
                for i in range(m):
                    delta = -(inverse[:, indices[i]] @ multipliers[i]).reshape(
                        m + 1, next_units)
                    keep = all_indices[all_indices != i]
                    delta_weights, delta_bias = delta[keep].T, delta[-1]
                    if scalar_output:
                        delta_weights, delta_bias = delta_weights[0], delta_bias[0]
                    results.append(GroupedFisherCompensation(
                        keep, delta_weights, delta_bias, float(costs[i]),
                        parameter_count - next_units, float(rcond)))
                return results

    # Removing feature i downdates the shared Gram matrix by the small block
    # U_i U_i^T, so Woodbury reduces every leave-one-out solve to its output width.
    kernel = (J @ J.T) * (X @ X.T)
    kernel *= scale[:, None] * scale[None, :]
    factor, info = torch.linalg.cholesky_ex((kernel + kernel.T) / 2)
    removed_blocks = scale[:, None, None] * A[:, :, None] * J[:, None, :]
    if int(info) == 0:
        solved_blocks = torch.cholesky_solve(
            removed_blocks.reshape(n, m * next_units), factor
        ).reshape(n, m, next_units)
        rank = n
    else:
        values, vectors = torch.linalg.eigh((kernel + kernel.T) / 2)
        cutoff = values[-1].clamp_min(0) * rcond ** 2
        kept = values > cutoff
        if not bool(kept.any()):
            base_weights = outgoing_weights[0] if scalar_output else outgoing_weights
            return [grouped_fisher_compensation(
                activations, fisher_weights, base_weights, i,
                downstream_jacobian, adjustable_mask, rcond) for i in range(m)]
        projection = vectors[:, kept].T @ removed_blocks.reshape(
            n, m * next_units)
        solved_blocks = (vectors[:, kept] @ (
            projection / values[kept, None])).reshape(n, m, next_units)
        rank = int(kept.sum())

    leverage = torch.einsum("nmi,nmj->mij", removed_blocks, solved_blocks)
    complement = torch.eye(next_units, dtype=A.dtype, device=A.device)[None] - leverage
    complement = (complement + complement.transpose(1, 2)) / 2
    small_factor, small_info = torch.linalg.cholesky_ex(complement)
    complement_spectrum = torch.linalg.eigvalsh(complement)
    good = ((small_info == 0)
            & (complement_spectrum[:, 0]
               > complement_spectrum[:, -1].clamp_min(0) * rcond ** 2))
    if m * next_units < rank:
        good[:] = False
    coefficients = torch.zeros(m, next_units, dtype=A.dtype, device=A.device)
    if bool(good.any()):
        coefficients[good] = torch.cholesky_solve(
            V.T[good].unsqueeze(-1), small_factor[good]).squeeze(-1)
    dual = torch.einsum("nmi,mi->nm", solved_blocks, coefficients)
    updates = torch.einsum("ni,nm,nj->mij", J, scale[:, None] * dual, X)

    targets = torch.einsum("nmi,mi->nm", removed_blocks, V.T)
    fitted = kernel @ dual
    removed_projection = torch.einsum(
        "nmi,mi->nm", removed_blocks,
        torch.einsum("nmi,nm->mi", removed_blocks, dual))
    costs = 0.5 * (targets - fitted + removed_projection).square().sum(0)

    all_indices = torch.arange(m, device=A.device)
    results = []
    for i in range(m):
        if not bool(good[i]):
            base_weights = outgoing_weights[0] if scalar_output else outgoing_weights
            results.append(grouped_fisher_compensation(
                activations, fisher_weights, base_weights, i,
                downstream_jacobian, adjustable_mask, rcond))
            continue
        keep = all_indices[all_indices != i]
        delta_weights = updates[i, :, keep]
        delta_bias = updates[i, :, -1]
        if scalar_output:
            delta_weights = delta_weights[0]
            delta_bias = delta_bias[0]
        results.append(GroupedFisherCompensation(
            survivor_indices=keep,
            delta_weights=delta_weights,
            delta_bias=delta_bias,
            predicted_cost=float(costs[i]),
            rank=rank,
            rcond=float(rcond),
        ))
    return results


def quantize_grouped_fisher_compensation(
        fit: GroupedFisherCompensation,
        activations: torch.Tensor,
        fisher_weights: torch.Tensor,
        outgoing_weights: torch.Tensor,
        remove_index,
        gamma: float,
        downstream_jacobian: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None) -> GroupedFisherCompensation:
    """Quantize compensation groups and reevaluate their joint Fisher cost."""
    A = activations.detach().to(dtype=torch.float64)
    q = fisher_weights.detach().to(dtype=torch.float64).clamp_min(0)
    scalar_output = outgoing_weights.ndim == 1
    V = (outgoing_weights.unsqueeze(0) if scalar_output else outgoing_weights)
    V = V.detach().to(dtype=torch.float64)
    n = len(A)
    if downstream_jacobian is None:
        downstream_jacobian = torch.ones(n, 1, dtype=A.dtype, device=A.device)
    J = downstream_jacobian.detach().to(dtype=torch.float64)
    remove = torch.as_tensor(remove_index, device=A.device, dtype=torch.long).reshape(-1)
    target = (A[:, remove] * (J @ V[:, remove])).sum(1)
    scale = (q / n).sqrt()
    step = gamma * torch.linalg.vector_norm(scale * target)

    weights = fit.delta_weights.unsqueeze(0) if scalar_output else fit.delta_weights
    bias = fit.delta_bias.reshape(1) if scalar_output else fit.delta_bias
    weights = weights.detach().to(dtype=torch.float64).clone()
    bias = bias.detach().to(dtype=torch.float64).clone()
    if group_ids is None:
        groups = torch.arange(len(fit.survivor_indices), device=A.device)
    else:
        groups = group_ids.to(device=A.device)[fit.survivor_indices]

    survivor_acts = A[:, fit.survivor_indices]
    total_groups = 0
    active_groups = 0
    if groups.numel():
        _, inverse = torch.unique(groups, return_inverse=True)
        total_groups = int(inverse.max()) + 1
        column_effects = survivor_acts * (J @ weights)
        grouped_effects = column_effects.new_zeros(n, int(inverse.max()) + 1)
        grouped_effects.scatter_add_(1, inverse.expand(n, -1), column_effects)
        magnitudes = torch.linalg.vector_norm(scale[:, None] * grouped_effects, dim=0)
        if float(step) == 0.0:
            factors = (magnitudes > 0).to(magnitudes.dtype)
        else:
            levels = step * torch.round(magnitudes / step)
            factors = torch.where(magnitudes > 0, levels / magnitudes, 0)
        active_groups = int((factors != 0).sum())
        weights *= factors[inverse][None]

    bias_magnitude = torch.linalg.vector_norm(scale * (J @ bias))
    if float(bias_magnitude) == 0.0:
        bias.zero_()
    elif float(step) != 0.0:
        bias *= step * torch.round(bias_magnitude / step) / bias_magnitude
    bias_active = bool(torch.count_nonzero(bias))
    if active_groups == 0:
        quantization_case = "bias_only" if bias_active else "off"
    else:
        quantization_case = "weights_and_bias" if bias_active else "weights_only"

    compensated = (survivor_acts * (J @ weights)).sum(1) + J @ bias
    cost = 0.5 * ((compensated - target) * scale).square().sum()
    delta_weights = weights[0] if scalar_output else weights
    delta_bias = bias[0] if scalar_output else bias
    return GroupedFisherCompensation(
        fit.survivor_indices, delta_weights, delta_bias, float(cost),
        fit.rank, fit.rcond, float(step), active_groups, total_groups,
        bias_active, quantization_case)


def quantize_grouped_fisher_singleton_compensations(
        fits: list[GroupedFisherCompensation],
        activations: torch.Tensor,
        fisher_weights: torch.Tensor,
        outgoing_weights: torch.Tensor,
        gamma: float,
        downstream_jacobian: torch.Tensor | None = None,
        remove_indices: torch.Tensor | None = None,
        ) -> list[GroupedFisherCompensation]:
    """Quantize all singleton-deletion fits in candidate batches."""
    A = activations.detach().to(dtype=torch.float64)
    q = fisher_weights.detach().to(dtype=torch.float64).clamp_min(0)
    scalar_output = outgoing_weights.ndim == 1
    V = outgoing_weights.unsqueeze(0) if scalar_output else outgoing_weights
    V = V.detach().to(dtype=torch.float64)
    n, m = A.shape
    remove = (torch.arange(m, device=A.device) if remove_indices is None else
              remove_indices.to(device=A.device, dtype=torch.long).reshape(-1))
    if len(fits) != len(remove):
        raise ValueError("fits and remove_indices must have the same length")
    if downstream_jacobian is None:
        downstream_jacobian = torch.ones(n, 1, dtype=A.dtype, device=A.device)
    J = downstream_jacobian.detach().to(dtype=torch.float64)
    scale = (q / n).sqrt()
    targets = A * (J @ V)
    steps = gamma * torch.linalg.vector_norm(scale[:, None] * targets, dim=0)
    chunk_size = max(1, min(len(fits), 4_000_000 // max(n * m, 1)))
    results = []

    for start in range(0, len(fits), chunk_size):
        stop = min(start + chunk_size, len(fits))
        chunk_fits = fits[start:stop]
        c = len(chunk_fits)
        survivor_indices = torch.stack([
            fit.survivor_indices.to(device=A.device) for fit in chunk_fits])
        weights = torch.zeros(c, V.shape[0], m, dtype=A.dtype, device=A.device)
        packed_weights = torch.stack([
            fit.delta_weights.unsqueeze(0) if scalar_output else fit.delta_weights
            for fit in chunk_fits]).to(dtype=A.dtype, device=A.device)
        weights.scatter_(2, survivor_indices[:, None].expand(-1, V.shape[0], -1),
                         packed_weights)
        bias = torch.stack([
            fit.delta_bias.reshape(1) if scalar_output else fit.delta_bias
            for fit in chunk_fits]).to(dtype=A.dtype, device=A.device)

        effects = A[None] * torch.einsum("nr,crm->cnm", J, weights)
        magnitudes = torch.linalg.vector_norm(scale[None, :, None] * effects, dim=1)
        chunk_remove = remove[start:stop]
        candidate_steps = steps[chunk_remove]
        safe_steps = candidate_steps.clamp_min(torch.finfo(A.dtype).tiny)
        levels = candidate_steps[:, None] * torch.round(
            magnitudes / safe_steps[:, None])
        levels = torch.where(candidate_steps[:, None] == 0, magnitudes, levels)
        factors = torch.where(magnitudes > 0, levels / magnitudes, 0)
        weights *= factors[:, None]
        effects *= factors[:, None]

        bias_effects = torch.einsum("nr,cr->cn", J, bias)
        bias_magnitudes = torch.linalg.vector_norm(scale[None] * bias_effects, dim=1)
        bias_levels = candidate_steps * torch.round(bias_magnitudes / safe_steps)
        bias_levels = torch.where(candidate_steps == 0, bias_magnitudes, bias_levels)
        bias_factors = torch.where(bias_magnitudes > 0,
                                   bias_levels / bias_magnitudes, 0)
        bias *= bias_factors[:, None]
        bias_effects *= bias_factors[:, None]

        residual = effects.sum(2) + bias_effects - targets[:, chunk_remove].T
        costs = 0.5 * (residual * scale[None]).square().sum(1)
        active_groups = (factors != 0).sum(1)
        bias_active = torch.count_nonzero(bias, dim=1) != 0

        for k, fit in enumerate(chunk_fits):
            dw = weights[k, :, fit.survivor_indices]
            db = bias[k]
            if scalar_output:
                dw, db = dw[0], db[0]
            active = int(active_groups[k])
            has_bias = bool(bias_active[k])
            case = ("bias_only" if has_bias else "off") if active == 0 else (
                "weights_and_bias" if has_bias else "weights_only")
            results.append(GroupedFisherCompensation(
                fit.survivor_indices, dw, db, float(costs[k]), fit.rank, fit.rcond,
                float(candidate_steps[k]), active, m - 1, has_bias, case))
    return results


def quantize_shortlisted_fisher_moves(moves: list[Move], gamma: float) -> list[Move]:
    prepared = list(moves)
    groups = {}
    for position, move in enumerate(moves):
        context = move.payload.get("_quant_context")
        if context is not None and move.payload.get("_quant_singleton"):
            groups.setdefault(id(context), (context, []))[1].append((position, move))

    quantized = {}
    for context, entries in groups.values():
        fits = [move.payload["_quant_fit"] for _, move in entries]
        remove = torch.tensor([move.payload["_quant_remove"] for _, move in entries],
                              device=context["activations"].device)
        outputs = quantize_grouped_fisher_singleton_compensations(
            fits, context["activations"], context["fisher_weights"],
            context["outgoing"], gamma, context["jacobian"], remove)
        quantized.update({position: fit for (position, _), fit in zip(entries, outputs)})

    for position, move in enumerate(moves):
        context = move.payload.get("_quant_context")
        if context is None:
            continue
        fit = quantized.get(position)
        if fit is None:
            fit = quantize_grouped_fisher_compensation(
                move.payload["_quant_fit"], context["activations"],
                context["fisher_weights"], context["outgoing"],
                move.payload["_quant_remove"], gamma, context["jacobian"],
                move.payload.get("_quant_group_ids"))
        payload = {k: v for k, v in move.payload.items() if not k.startswith("_quant_")}
        payload.update(delta_out=fit.delta_weights, delta_bias=fit.delta_bias,
                       gamma=gamma, fisher_cost=fit.predicted_cost,
                       fisher_rank=fit.rank, fisher_rcond=fit.rcond,
                       **quantization_metadata(fit))
        prepared[position] = Move(move.kind, move.site, move.layer, payload,
                                  pred_cost=fit.predicted_cost)
    return prepared


def quantization_metadata(fit: GroupedFisherCompensation):
    return {
        "quant_case": fit.quantization_case,
        "quant_weight_groups": fit.quantized_weight_groups,
        "quant_total_groups": fit.total_weight_groups,
        "quant_bias": fit.quantized_bias,
        "quant_step": fit.quantization_step,
    }


def grouped_fisher_pair_compensations(
        activations: torch.Tensor,
        fisher_weights: torch.Tensor,
        outgoing_weights: torch.Tensor,
        downstream_jacobian: torch.Tensor | None = None,
        adjustable_mask: torch.Tensor | None = None,
        candidate_indices: torch.Tensor | None = None,
        limit=None) -> list[tuple[torch.Tensor, GroupedFisherCompensation]]:
    """Score every pair and materialise the best joint compensations."""
    n, m = activations.shape
    if m < 2:
        return []
    candidates = (torch.arange(m, device=activations.device) if candidate_indices is None
                  else candidate_indices.to(device=activations.device, dtype=torch.long))
    pair_slots = torch.combinations(torch.arange(len(candidates), device=activations.device), 2)
    pairs = candidates[pair_slots]
    if limit is None:
        limit = len(pairs)
    limit = min(int(limit), len(pairs))
    scalar_output = outgoing_weights.ndim == 1
    V0 = outgoing_weights.unsqueeze(0) if scalar_output else outgoing_weights
    if adjustable_mask is not None and not bool(adjustable_mask.all()):
        base = outgoing_weights
        fits = [(pair, grouped_fisher_compensation(
            activations, fisher_weights, base, pair, downstream_jacobian,
            adjustable_mask)) for pair in pairs]
        return sorted(fits, key=lambda item: item[1].predicted_cost)[:limit]

    A = activations.detach().to(dtype=torch.float64)
    q = fisher_weights.detach().to(dtype=torch.float64).clamp_min(0)
    V = V0.detach().to(dtype=torch.float64)
    r = V.shape[0]
    if downstream_jacobian is None:
        if r != 1:
            raise ValueError("downstream_jacobian is required before a hidden next layer")
        downstream_jacobian = torch.ones(n, 1, dtype=A.dtype, device=A.device)
    J = downstream_jacobian.detach().to(dtype=torch.float64)
    original_r = r
    rcond = float(torch.finfo(activations.dtype).eps * max(n, r * (m + 1)))
    Uj, sj, Vhj = torch.linalg.svd(J, full_matrices=False)
    if sj.numel() == 0 or float(sj[0]) == 0.0:
        all_features = torch.arange(m, device=A.device)
        results = []
        for pair in pairs[:limit]:
            keep = all_features[(all_features != pair[0]) & (all_features != pair[1])]
            dw = A.new_zeros(original_r, len(keep))
            db = A.new_zeros(original_r)
            if scalar_output:
                dw, db = dw[0], db[0]
            results.append((pair, GroupedFisherCompensation(
                keep, dw, db, 0.0, 0, rcond)))
        return results
    r = max(int((sj > sj[0] * rcond).sum()), 1)
    lift = Vhj[:r].T
    J = Uj[:, :r] * sj[:r]
    V = Vhj[:r] @ V
    X = torch.cat((A, torch.ones(n, 1, dtype=A.dtype, device=A.device)), 1)
    scale = (q / n).sqrt()
    parameter_count = (m + 1) * r

    if parameter_count <= n:
        design = (X[:, :, None] * J[:, None, :]).reshape(n, parameter_count)
        design *= scale[:, None]
        gram = design.T @ design
        factor, info = torch.linalg.cholesky_ex((gram + gram.T) / 2)
        spectrum = torch.linalg.eigvalsh((gram + gram.T) / 2)
        well_conditioned = bool(spectrum[0] > spectrum[-1].clamp_min(0) * rcond ** 2)
        if int(info) == 0 and well_conditioned:
            inverse = torch.cholesky_inverse(factor)
            offsets = torch.arange(r, device=A.device)
            indices = (pairs[:, :, None] * r + offsets).reshape(len(pairs), 2 * r)
            blocks = inverse[indices[:, :, None], indices[:, None, :]]
            block_factor, block_info = torch.linalg.cholesky_ex(
                (blocks + blocks.transpose(1, 2)) / 2)
            if bool((block_info == 0).all()):
                rhs = V.T[pairs].reshape(len(pairs), 2 * r)
                multipliers = torch.cholesky_solve(
                    rhs.unsqueeze(-1), block_factor).squeeze(-1)
                costs = 0.5 * (rhs * multipliers).sum(1)
                chosen = torch.argsort(costs)[:limit]
                results = []
                all_features = torch.arange(m, device=A.device)
                for k in chosen.tolist():
                    pair, idx = pairs[k], indices[k]
                    delta = -(inverse[:, idx] @ multipliers[k]).reshape(m + 1, r)
                    keep = all_features[(all_features != pair[0]) & (all_features != pair[1])]
                    dw, db = delta[keep].T, delta[-1]
                    dw, db = lift @ dw, lift @ db
                    if scalar_output:
                        dw, db = dw[0], db[0]
                    results.append((pair, GroupedFisherCompensation(
                        keep, dw, db, float(costs[k]), design.shape[1] - 2 * r, rcond)))
                return results

    kernel = (J @ J.T) * (X @ X.T)
    kernel *= scale[:, None] * scale[None, :]
    values, vectors = torch.linalg.eigh((kernel + kernel.T) / 2)
    cutoff = values[-1].clamp_min(0) * rcond ** 2
    kept = values > cutoff
    if not bool(kept.any()):
        base = outgoing_weights
        fits = [(pair, grouped_fisher_compensation(
            activations, fisher_weights, base, pair, downstream_jacobian,
            adjustable_mask)) for pair in pairs]
        return sorted(fits, key=lambda item: item[1].predicted_cost)[:limit]

    # A Woodbury downdate cannot be inverted when the surviving basis has
    # fewer columns than the sampled Fisher kernel has live directions.
    if (m - 1) * r < int(kept.sum()):
        base = outgoing_weights
        fits = [(pair, grouped_fisher_compensation(
            activations, fisher_weights, base, pair, downstream_jacobian,
            adjustable_mask)) for pair in pairs]
        return sorted(fits, key=lambda item: item[1].predicted_cost)[:limit]

    removed = scale[:, None, None] * A[:, candidates, None] * J[:, None, :]
    projection = vectors[:, kept].T @ removed.reshape(n, len(candidates) * r)
    solved = (vectors[:, kept] @ (projection / values[kept, None])).reshape(
        n, len(candidates), r)
    feasible = []
    failed = []
    eye = torch.eye(2 * r, dtype=A.dtype, device=A.device)
    leverage = torch.einsum("nmi,nmj->mij", removed, solved)
    largest = torch.linalg.eigvalsh((leverage + leverage.transpose(1, 2)) / 2)[:, -1]
    if (len(candidates) >= 2
            and float(1.0 - torch.topk(largest, 2).values.sum()) > rcond ** 2):
        feasible = list(range(len(pairs)))
    else:
        chunk_size = max(1, 8_000_000 // max(n * 2 * r + 4 * r * r, 1))
        for start in range(0, len(pairs), chunk_size):
            chunk = pair_slots[start:start + chunk_size]
            blocks = removed[:, chunk].reshape(n, len(chunk), 2 * r)
            solved_blocks = solved[:, chunk].reshape(n, len(chunk), 2 * r)
            complement = eye[None] - torch.einsum("nbi,nbj->bij", blocks, solved_blocks)
            complement = (complement + complement.transpose(1, 2)) / 2
            _, status = torch.linalg.cholesky_ex(complement)
            spectrum = torch.linalg.eigvalsh(complement)
            good = ((status == 0)
                    & (spectrum[:, 0] > spectrum[:, -1].clamp_min(0) * rcond ** 2))
            feasible.extend(start + i for i in good.nonzero().flatten().tolist())
            failed.extend(start + i for i in (~good).nonzero().flatten().tolist())

    costs = {}
    chunk_size = max(1, 4_000_000 // max(n * 2 * r, 1))
    for start in range(0, len(feasible), chunk_size):
        chosen = torch.tensor(feasible[start:start + chunk_size],
                              dtype=torch.long, device=A.device)
        slots = pair_slots[chosen]
        blocks = removed[:, slots].reshape(n, len(chosen), 2 * r)
        solved_blocks = solved[:, slots].reshape(n, len(chosen), 2 * r)
        complement = eye[None] - torch.einsum(
            "nbi,nbj->bij", blocks, solved_blocks)
        rhs = V.T[pairs[chosen]].reshape(len(chosen), 2 * r)
        coefficients = torch.linalg.solve(
            (complement + complement.transpose(1, 2)) / 2, rhs.unsqueeze(-1)
        ).squeeze(-1)
        dual = torch.einsum("nbi,bi->nb", solved_blocks, coefficients)
        targets = torch.einsum("nbi,bi->nb", blocks, rhs)
        removed_projection = torch.einsum(
            "nbi,bi->nb", blocks, torch.einsum("nbi,nb->bi", blocks, dual))
        residual = targets - kernel @ dual + removed_projection
        for local, k in enumerate(chosen.tolist()):
            costs[k] = float(0.5 * residual[:, local].square().sum())

    base = outgoing_weights
    fallback = {k: grouped_fisher_compensation(
        activations, fisher_weights, base, pairs[k], downstream_jacobian,
        adjustable_mask) for k in failed}
    costs.update({k: fit.predicted_cost for k, fit in fallback.items()})
    selected = sorted(costs, key=costs.get)[:limit]

    all_features = torch.arange(m, device=A.device)
    materialized = {}
    good_selected = [k for k in selected if k not in fallback]
    for start in range(0, len(good_selected), chunk_size):
        chosen = torch.tensor(good_selected[start:start + chunk_size],
                              dtype=torch.long, device=A.device)
        slots = pair_slots[chosen]
        chosen_pairs = pairs[chosen]
        blocks = removed[:, slots].reshape(n, len(chosen), 2 * r)
        solved_blocks = solved[:, slots].reshape(n, len(chosen), 2 * r)
        complement = eye[None] - torch.einsum(
            "nbi,nbj->bij", blocks, solved_blocks)
        rhs = V.T[chosen_pairs].reshape(len(chosen), 2 * r)
        coefficients = torch.linalg.solve(
            (complement + complement.transpose(1, 2)) / 2, rhs.unsqueeze(-1)
        ).squeeze(-1)
        dual = torch.einsum("nbi,bi->nb", solved_blocks, coefficients)
        delta = torch.einsum("ni,nb,nj->bij", J, scale[:, None] * dual, X)
        targets = torch.einsum("nbi,bi->nb", blocks, rhs)
        removed_projection = torch.einsum(
            "nbi,bi->nb", blocks, torch.einsum("nbi,nb->bi", blocks, dual))
        residual = targets - kernel @ dual + removed_projection
        costs = 0.5 * residual.square().sum(0)
        for local, k in enumerate(chosen.tolist()):
            pair = pairs[k]
            keep = all_features[(all_features != pair[0]) & (all_features != pair[1])]
            dw, db = delta[local, :, keep], delta[local, :, -1]
            dw, db = lift @ dw, lift @ db
            if scalar_output:
                dw, db = dw[0], db[0]
            materialized[k] = GroupedFisherCompensation(
                keep, dw, db, float(costs[local]), int(kept.sum()), rcond)

    results = [(pairs[k], fallback[k] if k in fallback else materialized[k])
               for k in selected]
    return sorted(results, key=lambda item: item[1].predicted_cost)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def make_moons_data(n=2500, noise=0.25, seed=123, val_frac=0.2, test_frac=0.2):
    """Train / validation / test.  The scaler is fit on TRAIN only."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_frac,
                                                random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    t = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    l = lambda a: torch.tensor(a, dtype=torch.float32)
    data = {"train": (t(X_tr), l(y_tr)), "val": (t(X_va), l(y_va)), "test": (t(X_te), l(y_te))}
    return data, sc


def moons_fresh_sample(scaler, n=8000, noise=0.25, seed=999):
    """Fresh draw from the same distribution (for expected-information diagnostics)."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return (torch.tensor(scaler.transform(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class MLPAdapter(Adapter):
    move_groups = MOVE_GROUPS

    def __init__(self, data, widths=(2, 50, 50, 50, 1), step_layers=(0,)):
        super().__init__(data)
        self.widths = list(widths)
        self.L = len(widths) - 1
        self.loss_type = "bernoulli" if widths[-1] == 1 else "categorical"
        self.prunable = [f"W{l}" for l in range(self.L)]
        self.site_layer = {f"W{l}": l for l in range(self.L)}
        # layers whose INPUTS change sign (standardised features): only there is
        # a single weight -> +-inf a Heaviside of one input rather than a constant
        self.step_layers = tuple(step_layers)

    # ---- construction -------------------------------------------------------
    def init_params(self, seed=123) -> dict:
        g = torch.Generator().manual_seed(seed)
        P = {}
        for l in range(self.L):
            fan_in, fan_out = self.widths[l], self.widths[l + 1]
            bound = 1.0 / np.sqrt(fan_in)
            P[f"W{l}"] = (torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound
            P[f"b{l}"] = (torch.rand(fan_out, generator=g) * 2 - 1) * bound
        return P

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        struct = {}
        for l in range(self.L - 1):
            w = self.widths[l + 1]
            struct[f"type{l}"] = torch.full((w,), ACTIVE, dtype=torch.int8)
            struct[f"step_j{l}"] = torch.zeros(w, dtype=torch.long)
            struct[f"step_s{l}"] = torch.ones(w)
        return State(params, masks, struct)

    # ---- forward --------------------------------------------------------------
    def logits(self, P, S, X, return_acts=False):
        h = X
        acts = []
        out = None
        for l in range(self.L):
            z = h @ P[f"W{l}"].T + P[f"b{l}"]
            if l < self.L - 1:
                a = torch.sigmoid(z)
                t = S[f"type{l}"]
                if l in self.step_layers:
                    stepv = (h[..., S[f"step_j{l}"]] * S[f"step_s{l}"] > 0).to(a.dtype)
                    a = torch.where(t == STEP, stepv, a)
                a = a * (t != REMOVED).to(a.dtype)
                acts.append(a)
                h = a
            else:
                out = z.squeeze(-1) if self.widths[-1] == 1 else z
        return (out, acts) if return_acts else out

    @torch.no_grad()
    def next_preactivation_jacobians(self, s: State, acts=None,
                                     output_jacobian=None) -> list[torch.Tensor]:
        """Return a scalar output-direction Jacobian at every next preactivation."""
        X, _ = self.data["train"]
        P = self.effective(s)
        if acts is None:
            _, acts = self.logits(P, s.struct, X, return_acts=True)
        jac_z = [None] * self.L
        n = acts[0].shape[0]
        jac_z[self.L - 1] = (torch.ones(n, self.widths[-1], dtype=acts[0].dtype,
                                       device=acts[0].device)
                                 if output_jacobian is None else output_jacobian)
        for r in range(self.L - 2, 0, -1):
            active = (s.struct[f"type{r}"] == ACTIVE).to(acts[r].dtype)
            local = acts[r] * (1.0 - acts[r]) * active
            jac_z[r] = (jac_z[r + 1] @ P[f"W{r + 1}"]) * local
        return [jac_z[l + 1] for l in range(self.L - 1)]

    # ---- structural bookkeeping ----------------------------------------------
    def _kill(self, s: State, l: int, i: int) -> None:
        device = s.params[f"W{l}"].device
        zero_entries(s, f"W{l}", (torch.arange(self.widths[l], device=device) + i * self.widths[l]))
        zero_entries(s, f"b{l}", torch.tensor([i], device=device))
        Wn = f"W{l + 1}"
        col = torch.zeros_like(s.masks[Wn])
        col[:, i] = True
        zero_entries(s, Wn, col)
        s.struct[f"type{l}"][i] = REMOVED

    def _fold_const(self, s: State, l: int, i: int, c: float) -> None:
        Wn, bn = f"W{l + 1}", f"b{l + 1}"
        s.params[bn] += s.params[Wn][:, i] * s.masks[Wn][:, i].to(s.params[Wn].dtype) * c
        self._kill(s, l, i)

    def canonicalize(self, s: State) -> None:
        changed = True
        while changed:
            changed = False
            for l in range(self.L - 1):
                t = s.struct[f"type{l}"]
                fan_out = s.masks[f"W{l + 1}"].any(0)
                fan_in = s.masks[f"W{l}"].any(1)
                alive = t != REMOVED
                for i in (alive & ~fan_out).nonzero().flatten().tolist():
                    self._kill(s, l, i)
                    changed = True
                t = s.struct[f"type{l}"]
                const = (t == ACTIVE) & ~fan_in & s.masks[f"W{l + 1}"].any(0)
                for i in const.nonzero().flatten().tolist():
                    b = s.params[f"b{l}"][i] if bool(s.masks[f"b{l}"][i]) else s.params[f"b{l}"].new_zeros(())
                    self._fold_const(s, l, i, float(torch.sigmoid(b)))
                    changed = True

    # ---- structural moves -------------------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X, _ = self.data["train"]
        logits, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        ctx = {"acts": acts, "act_mean": [a.mean(0) for a in acts],
               "merge_pairs": [], "ablation_shortlist": [], "fisher_moves": []}
        enabled = self.resolve_struct_kinds(cfg.struct_kinds)
        merge_enabled = "unit_merge" in enabled
        for l, a in enumerate(acts):
            t = s.struct[f"type{l}"]
            ids = (t == ACTIVE).nonzero().flatten()
            pairs = []
            if merge_enabled and ids.numel() >= 2 and cfg.n_merge > 0:
                A = a[:, ids]
                Ac = A - A.mean(0)
                sd = Ac.pow(2).mean(0).sqrt().clamp_min(1e-12)
                C = (Ac.T @ Ac) / A.shape[0] / (sd[:, None] * sd[None, :])
                iu = torch.triu_indices(len(ids), len(ids), 1, device=ids.device)
                vals = C[iu[0], iu[1]].abs()
                top = torch.argsort(vals, descending=True)[: cfg.n_merge]
                for t_ in top.tolist():
                    p, q = int(iu[0, t_]), int(iu[1, t_])
                    for i, i2 in ((p, q), (q, p)):
                        xi, xj = Ac[:, i], Ac[:, i2]
                        alpha = float((xi * xj).mean() / xj.pow(2).mean().clamp_min(1e-12))
                        beta = float(A[:, i].mean() - alpha * A[:, i2].mean())
                        pairs.append((int(ids[i]), int(ids[i2]), alpha, beta, float(vals[t_])))
            ctx["merge_pairs"].append(pairs)

        if cfg.n_ablate > 0 and "unit_ablate" in enabled:
            if cfg.ablate_score == "gate_fisher":
                scores, signed = self.unit_gate_fisher(s)
            else:
                scores, signed = self.unit_write_energy(s, acts), None
            ranked = []
            for l, score in enumerate(scores):
                active = (s.struct[f"type{l}"] == ACTIVE).nonzero().flatten()
                ranked.extend((float(score[i]), l, int(i),
                               None if signed is None else float(signed[l][i]))
                              for i in active)
            ranked.sort(key=lambda item: item[0])
            ctx["ablation_shortlist"] = ranked[:cfg.n_ablate]

        if any(k in enabled for k in (
                "unit_fisher", "unit_fisher_pair", "unit_fisher_quantized",
                "unit_fisher_quantized_post")):
            if cfg.screen_n is not None and cfg.screen_n < logits.shape[0]:
                generator = torch.Generator(device=logits.device).manual_seed(cfg.seed + 7)
                idx = torch.randperm(logits.shape[0], generator=generator,
                                     device=logits.device)[:cfg.screen_n]
                fisher_logits = logits[idx]
                fisher_acts = [a[idx] for a in acts]
            else:
                fisher_logits, fisher_acts = logits, acts
            if self.widths[-1] == 1:
                probabilities = torch.sigmoid(fisher_logits)
                fisher_weights = probabilities * (1.0 - probabilities)
                jacobians = self.next_preactivation_jacobians(s, fisher_acts)
            else:
                probabilities = torch.softmax(fisher_logits, -1)
                generator = torch.Generator(device=fisher_logits.device).manual_seed(cfg.seed + 41)
                labels = torch.multinomial(probabilities, cfg.n_mc, replacement=True,
                                           generator=generator)
                score = (torch.nn.functional.one_hot(labels, self.widths[-1]).to(fisher_logits.dtype)
                         - probabilities[:, None, :]).reshape(-1, self.widths[-1])
                fisher_acts = [a.repeat_interleave(cfg.n_mc, 0) for a in fisher_acts]
                fisher_weights = torch.ones(score.shape[0], device=score.device,
                                             dtype=score.dtype)
                jacobians = self.next_preactivation_jacobians(
                    s, fisher_acts, output_jacobian=score)
            ctx["fisher_moves"] = self.unit_fisher_moves(
                s, fisher_acts, fisher_weights, jacobians,
                include_single="unit_fisher" in enabled,
                include_pairs="unit_fisher_pair" in enabled,
                pair_limit=cfg.n_struct * cfg.expand_on_fail,
                include_quantized="unit_fisher_quantized" in enabled,
                include_quantized_post="unit_fisher_quantized_post" in enabled,
                quant_gamma=cfg.fisher_quant_gamma)
        if "fisher_general" in enabled:
            from .fisher_general import graph_aware_fisher_moves
            ctx["fisher_moves"] = graph_aware_fisher_moves(self, s, cfg)
        return ctx

    @torch.no_grad()
    def unit_fisher_moves(self, s: State, acts=None, fisher_weights=None,
                          next_jacobians=None, include_single=True,
                          include_pairs=False, pair_limit=None,
                          include_quantized=False, include_quantized_post=False,
                          quant_gamma=0.2) -> list[Move]:
        """Grouped-Fisher candidates for active hidden units."""
        if self.L < 2:
            return []
        X, _ = self.data["train"]
        P = self.effective(s)
        if acts is None or fisher_weights is None:
            logits, acts = self.logits(P, s.struct, X, return_acts=True)
            if self.widths[-1] != 1:
                raise ValueError("categorical Fisher candidates require sampled score directions")
            probabilities = torch.sigmoid(logits)
            fisher_weights = probabilities * (1.0 - probabilities)
        if next_jacobians is None:
            next_jacobians = self.next_preactivation_jacobians(s, acts)
        moves = []
        for l in range(self.L - 1):
            out_name = f"W{l + 1}"
            live = ((s.struct[f"type{l}"] != REMOVED)
                    & s.masks[out_name].any(0)).nonzero().flatten()
            if live.numel() == 0:
                continue
            A = acts[l][:, live]
            next_ids = s.masks[out_name][:, live].any(1).nonzero().flatten()
            outgoing = P[out_name][next_ids][:, live]
            adjustable = s.masks[out_name][next_ids][:, live]
            next_jacobian = next_jacobians[l][:, next_ids]
            if include_single:
                fits = grouped_fisher_singleton_compensations(
                    A, fisher_weights, outgoing,
                    downstream_jacobian=next_jacobian,
                    adjustable_mask=adjustable,
                )
                for global_i_tensor, fit in zip(live, fits):
                    global_i = int(global_i_tensor)
                    if int(s.struct[f"type{l}"][global_i]) != ACTIVE:
                        continue
                    survivor_ids = live[fit.survivor_indices]
                    moves.append(Move(
                        "unit_fisher", f"h{l}", l,
                        {"l": l, "i": global_i, "survivor_ids": survivor_ids,
                         "next_ids": next_ids,
                         "delta_out": fit.delta_weights, "delta_bias": fit.delta_bias,
                         "fisher_cost": fit.predicted_cost, "fisher_rank": fit.rank,
                         "fisher_rcond": fit.rcond, "screen_cost": "fisher"},
                        pred_cost=fit.predicted_cost,
                    ))
            if include_quantized:
                fits = grouped_fisher_singleton_compensations(
                    A, fisher_weights, outgoing,
                    downstream_jacobian=next_jacobian,
                    adjustable_mask=adjustable,
                )
                quantized_fits = quantize_grouped_fisher_singleton_compensations(
                    fits, A, fisher_weights, outgoing, quant_gamma,
                    downstream_jacobian=next_jacobian)
                for global_i_tensor, quantized in zip(live, quantized_fits):
                    global_i = int(global_i_tensor)
                    if int(s.struct[f"type{l}"][global_i]) != ACTIVE:
                        continue
                    moves.append(Move(
                        "unit_fisher_quantized", f"h{l}", l,
                        {"l": l, "i": global_i,
                         "survivor_ids": live[quantized.survivor_indices],
                         "next_ids": next_ids,
                         "delta_out": quantized.delta_weights,
                         "delta_bias": quantized.delta_bias,
                         "gamma": quant_gamma,
                         "fisher_cost": quantized.predicted_cost,
                         "fisher_rank": quantized.rank,
                         "fisher_rcond": quantized.rcond,
                         "screen_cost": "fisher",
                         **quantization_metadata(quantized)},
                        pred_cost=quantized.predicted_cost))
            if include_quantized_post:
                fits = grouped_fisher_singleton_compensations(
                    A, fisher_weights, outgoing,
                    downstream_jacobian=next_jacobian,
                    adjustable_mask=adjustable,
                )
                context = {"activations": A, "fisher_weights": fisher_weights,
                           "outgoing": outgoing, "jacobian": next_jacobian}
                for local_i, (global_i_tensor, fit) in enumerate(zip(live, fits)):
                    global_i = int(global_i_tensor)
                    if int(s.struct[f"type{l}"][global_i]) != ACTIVE:
                        continue
                    moves.append(Move(
                        "unit_fisher_quantized_post", f"h{l}", l,
                        {"l": l, "i": global_i,
                         "survivor_ids": live[fit.survivor_indices],
                         "next_ids": next_ids,
                         "delta_out": fit.delta_weights,
                         "delta_bias": fit.delta_bias,
                         "fisher_cost": fit.predicted_cost,
                         "fisher_rank": fit.rank,
                         "fisher_rcond": fit.rcond,
                         "screen_cost": "fisher",
                         "_quant_context": context,
                         "_quant_fit": fit,
                         "_quant_remove": local_i,
                         "_quant_singleton": True},
                        pred_cost=fit.predicted_cost))
            if include_pairs:
                active_local = (s.struct[f"type{l}"][live] == ACTIVE).nonzero().flatten()
                pair_fits = grouped_fisher_pair_compensations(
                    A, fisher_weights, outgoing,
                    downstream_jacobian=next_jacobian,
                    adjustable_mask=adjustable, candidate_indices=active_local,
                    limit=pair_limit)
                for pair, fit in pair_fits:
                    global_pair = live[pair]
                    moves.append(Move(
                        "unit_fisher_pair", f"h{l}", l,
                        {"l": l, "i": int(global_pair[0]), "j": int(global_pair[1]),
                         "survivor_ids": live[fit.survivor_indices],
                         "next_ids": next_ids,
                         "delta_out": fit.delta_weights, "delta_bias": fit.delta_bias,
                         "fisher_cost": fit.predicted_cost, "fisher_rank": fit.rank,
                         "fisher_rcond": fit.rcond, "screen_cost": "fisher"},
                        pred_cost=fit.predicted_cost))
        return moves

    def prepare_structural_candidates(self, s: State, moves: list, cfg) -> list:
        return quantize_shortlisted_fisher_moves(moves, cfg.fisher_quant_gamma)

    @torch.no_grad()
    def unit_gate_fisher(self, s: State):
        """Per-unit empirical Fisher for a virtual multiplicative output gate.

        For example n and hidden unit i, q_ni = d loss_n / d gate_i.  The
        proposer score is E[q_ni^2]/2.  The signed first-order deletion estimate
        -E[q_ni] is returned for diagnostics.  A vectorised manual backward is
        used so all units are scored in one forward/backward-equivalent pass.
        """
        X, y = self.data["train"]
        P = self.effective(s)
        logits, acts = self.logits(P, s.struct, X, return_acts=True)
        # Derivative of per-example BCE-with-logits with respect to its logit.
        upstream = (torch.sigmoid(logits) - y.to(logits.dtype)).unsqueeze(1)
        scores = [None] * (self.L - 1)
        signed = [None] * (self.L - 1)
        for l in range(self.L - 2, -1, -1):
            W_next = P[f"W{l + 1}"]
            grad_a = upstream @ W_next
            gate_grad = acts[l] * grad_a
            scores[l] = 0.5 * gate_grad.square().mean(0)
            signed[l] = -gate_grad.mean(0)

            if l > 0:
                active = (s.struct[f"type{l}"] == ACTIVE).to(acts[l].dtype)
                # STEP outputs are nondifferentiable functions of the preceding
                # activation and REMOVED outputs are fixed at zero.  Only normal
                # sigmoid units transmit a derivative to the preceding layer.
                upstream = grad_a * acts[l] * (1.0 - acts[l]) * active
        return scores, signed

    @torch.no_grad()
    def unit_write_energy(self, s: State, acts=None):
        """Activation-only control: E[h_i^2] times outgoing-column norm."""
        if acts is None:
            X, _ = self.data["train"]
            _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        P = self.effective(s)
        return [a.square().mean(0) * P[f"W{l + 1}"].square().sum(0)
                for l, a in enumerate(acts)]

    def structural_moves(self, s: State, ctx: dict, cfg) -> list:
        moves = list(ctx.get("fisher_moves", []))
        for score, l, i, signed in ctx.get("ablation_shortlist", []):
            moves.append(Move("unit_ablate", f"h{l}", l,
                              {"l": l, "i": i, "gate_score": score,
                               "signed_taylor": signed}))
        for l in range(self.L - 1):
            t = s.struct[f"type{l}"]
            mean = ctx["act_mean"][l]
            for i in (t != REMOVED).nonzero().flatten().tolist():
                for c in (0.0, 1.0):
                    moves.append(Move("unit_sat", f"h{l}", l, {"l": l, "i": i, "c": c}))
                moves.append(Move("unit_mean", f"h{l}", l, {"l": l, "i": i, "c": float(mean[i])}))
            if l in self.step_layers:
                for i in (t == ACTIVE).nonzero().flatten().tolist():
                    X = self.data["train"][0] if l == 0 else ctx["acts"][l - 1]
                    valid = (X.amin(0) < 0) & (X.amax(0) > 0)
                    features = valid.nonzero().flatten()
                    if cfg.step_per_unit is not None and len(features) > cfg.step_per_unit:
                        scores = self.effective(s)[f"W{l}"][i, features].abs()
                        features = features[torch.topk(scores, cfg.step_per_unit).indices]
                    for j in features.tolist():
                        for sg in (1.0, -1.0):
                            moves.append(Move("unit_step", f"h{l}", l, {"l": l, "i": i, "j": j, "s": sg}))
            for (i, i2, alpha, beta, corr) in ctx["merge_pairs"][l]:
                moves.append(Move("unit_merge", f"h{l}", l,
                                  {"l": l, "i": i, "i2": i2, "alpha": alpha, "beta": beta, "corr": corr}))
        return moves

    def apply_structural(self, s: State, m: Move) -> bool:
        p = m.payload
        if m.kind == "fisher_general":
            for name, (indices, delta) in p["updates"].items():
                s.params[name].view(-1)[indices] += delta.to(s.params[name])
            for name, indices in p["deleted"].items():
                zero_entries(s, name, indices.to(s.params[name].device))
            return True
        l, i = p["l"], p["i"]
        t = s.struct[f"type{l}"]
        if m.kind == "unit_ablate":
            if int(t[i]) != ACTIVE:
                return False
            self._kill(s, l, i)
            return True
        if m.kind in ("unit_fisher", "unit_fisher_pair", "unit_fisher_quantized",
                      "unit_fisher_quantized_post"):
            removed = [i, p["j"]] if m.kind == "unit_fisher_pair" else [i]
            if any(int(t[j]) != ACTIVE for j in removed):
                return False
            survivor_ids = p["survivor_ids"]
            if survivor_ids.numel() and not bool((t[survivor_ids] != REMOVED).all()):
                return False
            Wn, bn = f"W{l + 1}", f"b{l + 1}"
            delta_out = p["delta_out"].to(s.params[Wn].dtype)
            delta_bias = p["delta_bias"].to(s.params[bn].dtype)
            next_ids = p.get("next_ids")
            if next_ids is None:
                s.params[Wn][:, survivor_ids] += delta_out
                s.params[bn] += delta_bias
            else:
                s.params[Wn][next_ids[:, None], survivor_ids[None]] += delta_out
                s.params[bn][next_ids] += delta_bias
            for j in removed:
                self._kill(s, l, j)
            return True
        if m.kind in ("unit_sat", "unit_mean"):
            if int(t[i]) == REMOVED:
                return False
            self._fold_const(s, l, i, p["c"])
            return True
        if m.kind == "unit_step":
            if int(t[i]) != ACTIVE or l not in self.step_layers:
                return False
            device = s.params[f"W{l}"].device
            zero_entries(s, f"W{l}", torch.arange(self.widths[l], device=device) + i * self.widths[l])
            zero_entries(s, f"b{l}", torch.tensor([i], device=device))
            t[i] = STEP
            s.struct[f"step_j{l}"][i] = p["j"]
            s.struct[f"step_s{l}"][i] = p["s"]
            return True
        if m.kind == "unit_merge":
            i2 = p["i2"]
            if int(t[i]) != ACTIVE or int(t[i2]) != ACTIVE:
                return False
            Wn, bn = f"W{l + 1}", f"b{l + 1}"
            W, M = s.params[Wn], s.masks[Wn]
            col_i = W[:, i] * M[:, i].to(W.dtype)
            W[:, i2] = W[:, i2] * M[:, i2].to(W.dtype) + p["alpha"] * col_i
            M[:, i2] = M[:, i2] | M[:, i]
            s.params[bn] += p["beta"] * col_i
            self._kill(s, l, i)
            return True
        raise ValueError(m.kind)

    # ---- classical structured baseline ------------------------------------------
    @torch.no_grad()
    def classical_unit_order(self, s: State) -> list:
        """Mean-ablation order: Var(a_i) * ||W_{l+1}[:, i]||^2, smallest first."""
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        items = []
        for l, a in enumerate(acts):
            Wn = s.params[f"W{l + 1}"] * s.masks[f"W{l + 1}"]
            sc = a.var(0) * Wn.pow(2).sum(0)
            t = s.struct[f"type{l}"]
            items += [(float(sc[i]), (l, i)) for i in range(a.shape[1]) if int(t[i]) != REMOVED]
        return [u for _, u in sorted(items, key=lambda t: t[0])]

    @torch.no_grad()
    def mean_move(self, s: State, unit) -> Move:
        l, i = unit
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        return Move("unit_mean", f"h{l}", l, {"l": l, "i": i, "c": float(acts[l][:, i].mean())})

    # ---- Wanda input norms --------------------------------------------------------
    @torch.no_grad()
    def input_norms(self, s: State) -> dict:
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        ins = [X] + acts
        return {f"W{l}": ins[l].pow(2).sum(0).sqrt()[None, :] for l in range(self.L)}


# --------------------------------------------------------------------------
# Training (same recipe as pruningNN.ipynb: Adam 1e-3, full batch)
# --------------------------------------------------------------------------

def train_mlp(adapter: MLPAdapter, epochs=2500, lr=1e-3, seed=123, verbose=True,
              batch_size=None):
    torch.manual_seed(seed)
    X, y = adapter.data["train"]
    P = {k: v.to(X.device).requires_grad_(True) for k, v in adapter.init_params(seed).items()}
    s = adapter.make_state(P).to(X.device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size or len(X),
                        shuffle=batch_size is not None)
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    hist = []
    for ep in range(epochs):
        total = torch.zeros((), device=X.device)
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = adapter.loss_from_logits(adapter.logits(P, s.struct, xb), yb).mean()
            loss.backward()
            opt.step()
            total += loss.detach() * len(xb)
        train_loss = float(total / len(X))
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                st = adapter.make_state(P).to(X.device)
                hist.append({"epoch": ep, "train_loss": train_loss,
                             "val_loss": adapter.metrics(st, "val")["loss"]})
            if verbose and ep % 500 == 0:
                print(f"epoch {ep:5d}  train {hist[-1]['train_loss']:.4f}  val {hist[-1]['val_loss']:.4f}")
    return adapter.make_state(P).to(X.device), hist
