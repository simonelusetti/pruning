from __future__ import annotations

import torch


@torch.no_grad()
def complement_update(weight, a, b, deleted, *, mu=0., rtol=1e-6,
                      max_iterations=1000, progress=None, b_inverse=None):
    """Minimize the AXB quadratic with fixed deletions and all survivors free."""
    if weight.ndim != 2 or deleted.shape != weight.shape or deleted.dtype != torch.bool:
        raise ValueError("weight and boolean deletion mask must have the same matrix shape")
    if mu < 0 or not 0 < rtol < 1 or max_iterations < 1:
        raise ValueError("invalid damping or convergence settings")
    if a.shape != (weight.shape[1],) * 2 or b.shape != (weight.shape[0],) * 2:
        raise ValueError("AXB factors do not match the weight matrix")
    a = a.to(device=weight.device, dtype=weight.dtype)
    b = b.to(device=weight.device, dtype=weight.dtype)
    a = (a + a.T) / 2
    b = (b + b.T) / 2
    deleted = deleted.to(weight.device)
    free = ~deleted
    fixed = torch.where(deleted, -weight, 0.)
    rhs = -(b @ fixed @ a).masked_fill(deleted, 0.)
    rhs_norm = torch.linalg.vector_norm(rhs)
    update = torch.zeros_like(weight)
    iterations = 0
    residual = rhs.clone()
    if float(rhs_norm) > 0:
        if progress is not None:
            progress(0, 1.)
        if mu == 0:
            a_inverse = torch.cholesky_inverse(torch.linalg.cholesky(a))
            if b_inverse is None:
                b_inverse = torch.cholesky_inverse(torch.linalg.cholesky(b))
            else:
                b_inverse = b_inverse.to(weight)

            def precondition(values):
                return (b_inverse @ values @ a_inverse).masked_fill(deleted, 0.)
        else:
            a_values, a_vectors = torch.linalg.eigh(a)
            b_values, b_vectors = torch.linalg.eigh(b)
            denominator = b_values[:, None] * a_values[None, :] + mu
            if float(a_values.min()) <= 0 or float(b_values.min()) <= 0:
                raise ValueError("complement updates require positive definite AXB factors")

            def precondition(values):
                spectral = b_vectors.T @ values @ a_vectors
                return (b_vectors @ (spectral / denominator) @ a_vectors.T).masked_fill(deleted, 0.)

        z = precondition(residual)
        direction = z.clone()
        rz = (residual * z).sum()
        for iterations in range(1, max_iterations + 1):
            product = (b @ direction @ a + mu * direction).masked_fill(deleted, 0.)
            curvature = (direction * product).sum()
            if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
                raise RuntimeError("complement CG encountered nonpositive curvature")
            alpha = rz / curvature
            update.add_(alpha * direction)
            residual.sub_(alpha * product)
            if progress is not None and iterations % 25 == 0:
                progress(iterations, float(torch.linalg.vector_norm(residual) / rhs_norm))
            if float(torch.linalg.vector_norm(residual)) <= rtol * float(rhs_norm):
                # A fresh residual guards against recursive CG residual drift.
                residual = rhs - (b @ update @ a + mu * update).masked_fill(deleted, 0.)
                if float(torch.linalg.vector_norm(residual)) <= rtol * float(rhs_norm):
                    break
                z = precondition(residual)
                direction = z.clone()
                rz = (residual * z).sum()
                continue
            z = precondition(residual)
            next_rz = (residual * z).sum()
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        residual = rhs - (b @ update @ a + mu * update).masked_fill(deleted, 0.)
        if float(torch.linalg.vector_norm(residual)) > rtol * float(rhs_norm):
            raise RuntimeError(
                f"complement CG did not converge in {max_iterations} iterations; "
                f"relative residual={float(torch.linalg.vector_norm(residual) / rhs_norm):g}")
    delta = fixed + update.masked_fill(deleted, 0.)
    damage = .5 * (delta * (b @ delta @ a)).sum()
    return delta, {
        "iterations": iterations,
        "relative_residual": float(torch.linalg.vector_norm(residual) / rhs_norm)
        if float(rhs_norm) > 0 else 0.,
        "predicted_fisher_damage": float(damage),
        "regularized_objective": float(damage + .5 * mu * update.square().sum()),
        "free_coordinates": int(free.sum()),
    }


def inverse_diagonal(matrix: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    """Diagonal of a PSD matrix inverse without materializing the inverse."""
    matrix = ((matrix + matrix.T) / 2).float()
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info):
        return torch.linalg.pinv(matrix.double(), hermitian=True).diag()
    result = torch.empty(len(matrix), dtype=torch.float64, device=matrix.device)
    for start in range(0, len(matrix), chunk_size):
        end = min(start + chunk_size, len(matrix))
        rhs = torch.zeros((len(matrix), end - start), dtype=matrix.dtype,
                          device=matrix.device)
        rhs[start:end].diagonal().fill_(1)
        columns = torch.linalg.solve_triangular(factor, rhs, upper=False)
        result[start:end] = columns.double().square().sum(0)
    return result
