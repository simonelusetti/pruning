"""
Vendored from Fdist_Pruning (Berman, Fu, Hirst, Obirai, "Optimal Pruning for
Neural Architectures using Fisher Information Distances", arXiv:2609.16129),
https://github.com/edhirst/Fdist_Pruning, commit 8d60ea99027a284f8f02834e8d7d5c72bc278cda,
file src/utils/fim_calculator.py.  Licensed under the Apache License 2.0; see
LICENSE-Fdist_Pruning in this folder.

Modifications: only the two functions used here are kept
(`calculate_fim_backprop`, `fisher_entries_forward`), copied verbatim; the
module-level `from nngeometry import ...` import and the other functions are
removed.  Nothing inside the two functions is changed.
"""
import torch
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call


def calculate_fim_backprop(model, loader, device="cpu", chunk_size=32):
    """
    Calculate the class-marginal MODEL Fisher diagonal in a SINGLE pass over
    the data, for ALL parameter tensors at once:

    The expectation over classes is taken against the model's own predictive
    distribution p(c|x), NOT against the observed labels, so this is the model
    Fisher and not the empirical Fisher. The distinction matters: only the former
    is the Fisher information of the model, and so only the former carries the
    geometric meaning the f_dist schemes rely on.


        F_ii = E_x[ sum_c p(c|x) * (d log p(c|x) / d theta_i)^2 ]

    Per batch chunk and per class c, one vmap'd per-sample gradient of
    log p(c|x) w.r.t. the full parameter dict accumulates p-weighted squared
    gradients for every tensor simultaneously. This replaces the previous
    per-tensor implementation (kept below as calculate_fim_backprop_per_tensor),
    which re-iterated the full dataloader once per parameter tensor — identical
    math and output, ~#tensors times faster (6x for SimpleNN, ~60x for SimpleViT).

    Args:
        model: PyTorch model returning logits [B, C]
        loader: DataLoader for the Fisher subset
        device: Device to run computation on
        chunk_size: per-sample-gradient chunk to bound memory (chunk x P floats)

    Returns:
        1D CPU tensor whose flatten order matches model.named_parameters().
    """
    model = model.to(device)
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    if not params:
        return torch.empty(0)

    def logp_one(pr, x_single, c_long):
        logits = functional_call(model, (pr, buffers), (x_single.unsqueeze(0),))
        logp = F.log_softmax(logits, dim=1)  # [1, C]
        return logp[0, c_long]

    grad_one = grad(logp_one)                       # d log p(c|x) / d params (dict)
    grad_batch = vmap(grad_one, in_dims=(None, 0, None))

    acc = {k: torch.zeros_like(v, device=device) for k, v in params.items()}
    seen = 0

    for xb, _ in loader:
        xb = xb.to(device)
        for x_chunk in xb.split(chunk_size):
            b = x_chunk.size(0)
            with torch.no_grad():
                logits = functional_call(model, (params, buffers), (x_chunk,))
                probs = torch.softmax(logits, dim=1)  # [b, C]
            C = probs.size(1)

            for c in range(C):
                c_long = torch.tensor(c, device=device, dtype=torch.long)
                gtree = grad_batch(params, x_chunk, c_long)      # dict of [b, ...param_shape]
                w = probs[:, c]
                for k, g in gtree.items():
                    acc[k] += (w.reshape(b, *([1] * (g.ndim - 1))) * g ** 2).sum(dim=0)

            seen += b

    parts = []
    for name, _ in model.named_parameters():
        parts.append((acc[name] / float(max(seen, 1))).reshape(-1).to("cpu"))

    return torch.cat(parts, dim=0)


def fisher_entries_forward(
    model,
    loader,
    tensor_name,
    local_idxs,
    alphas,
    device="cpu",
    probe_batch=32,
    sample_chunk=None,
):
    """
    Fisher diagonal entries F_ii evaluated at singly-perturbed parameter states.

    For each probe b the model is evaluated with
        params[tensor_name].view(-1)[local_idxs[b]] *= alphas[b]
    and every other parameter left untouched, then F_ii is returned for that
    same coordinate i = (tensor_name, local_idxs[b]).

    Args:
        model: PyTorch model returning logits [B, C]
        loader: DataLoader for the Fisher subset
        tensor_name: name (in model.named_parameters()) of the tensor holding
            every probed coordinate
        local_idxs: 1D LongTensor [Nb] of flat indices into that tensor
        alphas: 1D float tensor [Nb], the scale applied to each coordinate
        device: device to run on
        probe_batch: probes vmapped together per call. The batch dimension is
            what gives CPU intra-op threading something to parallelise, so
            raising it is the main throughput knob on many-core nodes. Memory is
            O(probe_batch * tensor.numel()).
        sample_chunk: samples vmapped together (default: the whole batch)

    Returns:
        1D CPU tensor [Nb] of Fisher diagonal entries.
    """
    model = model.to(device)
    model.eval()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    if tensor_name not in params:
        raise KeyError(f"{tensor_name!r} is not a parameter of this model")

    T = params[tensor_name]
    n = T.numel()
    other = {k: v for k, v in params.items() if k != tensor_name}

    local_idxs = torch.as_tensor(local_idxs, dtype=torch.long, device=device).reshape(-1)
    alphas = torch.as_tensor(alphas, dtype=T.dtype, device=device).reshape(-1)
    if local_idxs.numel() != alphas.numel():
        raise ValueError(f"local_idxs ({local_idxs.numel()}) and alphas ({alphas.numel()}) must match")
    if local_idxs.numel() and int(local_idxs.max()) >= n:
        raise IndexError(f"index {int(local_idxs.max())} out of range for {tensor_name} (numel {n})")

    def one_probe(T_pert, E_tan, x_single):
        def f(t):
            p = dict(other)
            p[tensor_name] = t
            return functional_call(model, (p, buffers), (x_single.unsqueeze(0),))[0]  # [C]
        logits, J = torch.func.jvp(f, (T_pert,), (E_tan,))
        pr = torch.softmax(logits, 0)
        return (pr * (J - (pr * J).sum()) ** 2).sum()          # Var_{c~p}[J_c]

    over_samples = vmap(one_probe, in_dims=(None, None, 0))
    over_probes = vmap(lambda Tp, Et, xs: over_samples(Tp, Et, xs).sum(), in_dims=(0, 0, None))

    nb = local_idxs.numel()
    acc = torch.zeros(nb, dtype=T.dtype, device=device)
    seen = 0

    for xb, _ in loader:
        xb = xb.to(device)
        chunks = xb.split(sample_chunk) if sample_chunk else [xb]
        for x_chunk in chunks:
            for s in range(0, nb, probe_batch):
                idx = local_idxs[s:s + probe_batch]
                al = alphas[s:s + probe_batch]
                b = idx.numel()

                Tp = T.reshape(1, n).repeat(b, 1)
                rows = torch.arange(b, device=device)
                Tp[rows, idx] = Tp[rows, idx] * al
                Et = torch.zeros(b, n, dtype=T.dtype, device=device)
                Et[rows, idx] = 1.0

                acc[s:s + b] += over_probes(Tp.view(b, *T.shape), Et.view(b, *T.shape), x_chunk)
            seen += x_chunk.size(0)

    return (acc / float(max(seen, 1))).detach().cpu()

