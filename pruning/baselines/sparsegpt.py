from __future__ import annotations

import torch


@torch.inference_mode()
def sparsegpt_matrix(weight: torch.Tensor, hessian: torch.Tensor, sparsity: float,
                     percdamp: float = .01, blocksize: int = 128,
                     output_inverse_diagonal: torch.Tensor | None = None):
    """SparseGPT's unstructured reconstruction kernel."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if sparsity == 0:
        return weight.clone(), torch.zeros_like(weight, dtype=torch.bool)

    dtype = weight.dtype
    values = weight.float().clone()
    curvature = hessian.to(device=weight.device, dtype=torch.float32).clone()
    dead = curvature.diag() == 0
    curvature[dead, dead] = 1
    values[:, dead] = 0
    curvature.diagonal().add_(percdamp * curvature.diag().mean())
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(curvature)), upper=True)
    pruned = torch.zeros_like(values, dtype=torch.bool)

    for start in range(0, values.shape[1], blocksize):
        end = min(start + blocksize, values.shape[1])
        current = values[:, start:end].clone()
        result = torch.zeros_like(current)
        errors = torch.zeros_like(current)
        local_inverse = inverse[start:end, start:end]
        metric = current.square() / local_inverse.diag()[None, :].square()
        if output_inverse_diagonal is not None:
            metric = metric / output_inverse_diagonal.to(metric)[:, None].clamp_min(1e-30)
        rank = min(int(metric.numel() * sparsity), metric.numel() - 1)
        mask = metric <= torch.kthvalue(metric.flatten(), rank + 1).values

        for column in range(end - start):
            vector = current[:, column]
            reconstructed = vector.clone()
            reconstructed[mask[:, column]] = 0
            result[:, column] = reconstructed
            error = (vector - reconstructed) / local_inverse[column, column]
            current[:, column:] -= error[:, None] @ local_inverse[
                column, column:][None, :]
            errors[:, column] = error

        values[:, start:end] = result
        pruned[:, start:end] = mask
        values[:, end:] -= errors @ inverse[start:end, end:]

    pruned[:, dead] = True
    values[pruned] = 0
    return values.to(dtype), pruned
