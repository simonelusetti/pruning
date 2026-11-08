"""Exact categorical Fisher blocks used by the modular-addition baseline."""
from __future__ import annotations
import math
import torch
NAMES = ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out", "W_U", "b_U")


def flat(tensors):
    return torch.cat([tensors[n].flatten() for n in NAMES])


def unflat(vector, like):
    out, offset = {}, 0
    for n in NAMES:
        out[n] = vector[offset:offset + like[n].numel()].view_as(like[n])
        offset += like[n].numel()
    return out


def get_flat(adapter, state):
    return flat(adapter.effective(state)).double(), flat(state.masks)


def set_flat(state, theta, live):
    for n, p in unflat(theta, state.params).items():
        state.params[n] = p.to(state.params[n].dtype).clone()
    for n, m in unflat(live, state.masks).items():
        state.masks[n] = m.clone()


def row_view(P):
    """[rows, width] per group; one row = the parameters feeding one output coordinate."""
    per_head = lambda W: W.permute(0, 2, 1).reshape(-1, W.shape[1])
    return {"W_E": P["W_E"], "W_pos": P["W_pos"],
            "W_Q": per_head(P["W_Q"]), "W_K": per_head(P["W_K"]), "W_V": per_head(P["W_V"]),
            "W_O": P["W_O"].permute(2, 0, 1).reshape(P["W_O"].shape[2], -1),
            "W_in": torch.cat((P["W_in"], P["b_in"][:, None]), 1),
            "W_out": torch.cat((P["W_out"], P["b_out"][:, None]), 1),
            "W_U": torch.cat((P["W_U"].T, P["b_U"][:, None]), 1)}


def sqrt_fisher(p):
    """Rows s_c = sqrt(p_c)(e_c - p): sum_c s_c s_c^T = diag p - p p^T."""
    return p.sqrt()[:, :, None] * (torch.eye(p.shape[1], dtype=p.dtype, device=p.device) - p[:, None, :])


@torch.no_grad()
def fisher_blocks(adapter, P, struct, X):
    """Sums over (x, c) of g g^T for every row of row_view and for the whole W_O, where
    g_{x,c} = s_c^T d logits / d theta.  One layer, logits read at the last position: each
    gradient row is (class-dependent vector) x (input), so most class sums collapse first."""
    logits, cache = adapter.forward_full(P, struct, X)
    s = sqrt_fisher(logits.softmax(-1))                                   # [b, c, logit]
    alive = struct["head_alive"].to(P["W_O"].dtype)
    free = (~struct["head_fixed"]).to(P["W_O"].dtype)                      # fixed patterns: no score path
    x0 = cache["x0"]                                                       # [b, t, d]
    q = torch.einsum("btd,hde->bhte", x0, P["W_Q"])[:, :, -1]               # [b, h, e]
    k = torch.einsum("btd,hde->bhte", x0, P["W_K"])
    v = torch.einsum("btd,hde->bhte", x0, P["W_V"])
    a = cache["attn"][:, :, -1]                                            # [b, h, t]
    z, x1, act, x2 = cache["z"][:, :, -1], cache["x1"][:, -1], cache["act"][:, -1], cache["x2"][:, -1]
    u2 = s @ P["W_U"].T                                                    # d/dx2
    gp = (u2 @ P["W_out"]) * (act > 0)[:, None]                            # d/d MLP preactivation
    u1 = u2 + gp @ P["W_in"]                                               # d/dx1
    gz = torch.einsum("bcd,hed->bche", u1, P["W_O"]) * alive[:, None]      # d/dz
    ga = torch.einsum("bche,bhte->bcht", gz, v)                            # d/d attention
    gs = a[:, None] * (ga - (a[:, None] * ga).sum(-1, keepdim=True)) * free[:, None]  # d/d score
    r = 1 / math.sqrt(adapter.dh)
    gq = torch.einsum("bcht,bhte->bche", gs, k) * r
    gx0 = (torch.einsum("bht,bche,hde->bctd", a, gz, P["W_V"])
           + torch.einsum("bcht,bhe,hde->bctd", gs, q, P["W_K"]) * r)
    gx0[:, :, -1] += u1 + torch.einsum("bche,hde->bcd", gq, P["W_Q"])
    ones = torch.ones(len(X), 1, dtype=x0.dtype, device=X.device)
    outer = lambda w, inp: torch.einsum("br,bi,bj->rij", w, inp, inp)     # rows sharing one input
    per_head = lambda w, inp: torch.einsum("bhe,bhi,bhj->heij", w, inp, inp).flatten(0, 1)
    tokens = torch.nn.functional.one_hot(X, P["W_E"].shape[0]).to(x0.dtype)
    E = torch.einsum("btv,bctd->bcvd", tokens, gx0)
    w_K = torch.einsum("bcht,btd->bchd", gs, x0)
    za = (z * alive[:, None]).flatten(1)                                   # dead heads: no W_O gradient
    rows = {
        "W_E": torch.einsum("bcvi,bcvj->vij", E, E),
        "W_pos": torch.einsum("bcti,bctj->tij", gx0, gx0),
        "W_Q": per_head(gq.square().sum(1), x0[:, -1, None].expand(-1, adapter.H, -1)),
        "W_K": torch.einsum("bhe,bchi,bchj->heij", (q * r).square(), w_K, w_K).flatten(0, 1),
        "W_V": per_head(gz.square().sum(1), torch.einsum("bht,btd->bhd", a, x0)),
        "W_O": outer(u1.square().sum(1), za),
        "W_in": outer(gp.square().sum(1), torch.cat((x1, ones), 1)),
        "W_out": outer(u2.square().sum(1), torch.cat((act, ones), 1)),
        "W_U": outer(s.square().sum(1), torch.cat((x2, ones), 1))}
    M = torch.einsum("bco,bcp->bop", u1, u1)
    F_O = torch.einsum("bop,bi,bj->oipj", M, za, za).flatten(0, 1).flatten(1, 2)
    return rows, F_O


def fisher(adapter, state, X, blocks=True, chunk=128):
    """Exact model Fisher: the row blocks of every group, the full W_O block, and the diagonal
    (taken from the row blocks, which partition theta)."""
    P0 = {k: v.double() for k, v in adapter.effective(state).items()}
    theta = flat(P0)
    idx = row_view(unflat(torch.arange(theta.numel(), device=theta.device), P0))
    rows, F_O = None, None
    for start in range(0, len(X), chunk):
        r, f = fisher_blocks(adapter, P0, state.struct, X[start:start + chunk])
        rows = r if rows is None else {g: rows[g] + r[g] for g in r}
        F_O = f if F_O is None else F_O + f
    n = len(X)
    rows = {g: F / n for g, F in rows.items()}
    diag = torch.zeros_like(theta)
    for g, I in idx.items():
        diag[I] = rows[g].diagonal(dim1=1, dim2=2)
    return diag, rows, F_O / n, idx


def berman_step(theta, live, diag, count):
    cost = (.5 * theta.square() * diag).masked_fill(~live, torch.inf)
    D = torch.zeros_like(live)
    D[torch.argsort(cost)[:min(count, int(live.sum()))]] = True
    return D
