"""
Numerical check of the exact LayerNorm route for the stream move, on the 4-block
pre-LN character model of ln_freeze_test.py (results_ln/gpt_char.pt).

In a pre-LN transformer every matrix that acts on the stream x comes after a
LayerNorm (ln1, ln2 of each block, final ln).  Notation: C = I - 11^T/d.

Step A (exact).  Multiply every term added to x by C (embedding, positions, attention
output W_o, b_o, MLP output W_2, b_2).  LN(Cx) = LN(x), so nothing changes, and now
x is always orthogonal to 1, so inside LN:  x - mean(x) = x,  sigma(x)^2 = |x|^2/d + eps.

Step B (the move).  Take a unit u orthogonal to 1.  At the input of every LayerNorm k,
replace u^T x by a constant m_k (its training mean at that LayerNorm):
     x -> Pi x + m_k u,   Pi = I - u u^T                          ('replaced network')
Then x lives in u-perp (d-1 numbers y = V^T x) and every LayerNorm k followed by a
matrix W (+ bias) computes, EXACTLY,
     W LN(x) + bias = [W diag(g) V] y / s  +  [m_k W diag(g) u] / s  +  [W b + bias],
     s = sqrt((|y|^2 + m_k^2)/d + eps)
i.e. a smaller matrix, a bias added BEFORE the division by s, the usual bias, and one
number m_k^2 inside the square root ('explicit network').

Checks: (1) Step A leaves the outputs unchanged; (2) replaced == explicit network;
(3) loss of the move for the best candidate directions, with position-independent
m_k and with one m_k per position, against freezing sigma at every LayerNorm.
Parameters: one direction frees 6576 numbers here; the pre-division biases cost
4*(384 + 512) + 180 = 3764 once (they accumulate over later directions).

Usage:  python ln_exact_route.py        (needs results_ln/gpt_char.pt; writes results_ln/exact_route.csv)
"""
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ln_freeze_test as LF  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
OUT = os.environ.get("LN_RESULTS", os.path.join(HERE, "results_ln"))
EPS = 1e-5


def params(model):
    g = lambda t: t.detach().double().clone()
    P = {"E": g(model.emb.weight), "P": g(model.pos), "gf": g(model.lnf.g), "bf": g(model.lnf.b),
         "Wh": g(model.head.weight), "bh": g(model.head.bias), "blocks": []}
    for b in model.blocks:
        P["blocks"].append({"g1": g(b.ln1.g), "b1": g(b.ln1.b), "Wqkv": g(b.qkv.weight), "bqkv": g(b.qkv.bias),
                            "Wo": g(b.o.weight), "bo": g(b.o.bias), "g2": g(b.ln2.g), "b2": g(b.ln2.b),
                            "W1": g(b.fc1.weight), "bf1": g(b.fc1.bias), "W2": g(b.fc2.weight), "bf2": g(b.fc2.bias)})
    return P


def attn(h, bp):
    B, T, _ = h.shape
    D, H = LF.D, LF.H
    q, k, v = h.view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return a.transpose(1, 2).reshape(B, T, D) @ bp["Wo"].T + bp["bo"]


def ln_std(x, g, b):
    c = x - x.mean(-1, keepdim=True)
    return g * c / torch.sqrt(c.pow(2).mean(-1, keepdim=True) + EPS) + b


def forward_ref(P, idx, replace=None, frozen=None, record=None):
    """Plain forward (standard LayerNorm).  replace: (u, m) with m a list of 9 tensors
    (per LayerNorm; shape () or (T,)) -> LN inputs x -> Pi x + m u.  frozen: list of 9
    (s, s0) -> 1/sigma replaced by constants.  record: list collecting LN inputs."""
    x = P["E"][idx] + P["P"][: idx.shape[1]]
    k = 0

    def LNk(x, g, b):
        nonlocal k
        if record is not None:
            record.append(x)
        if replace is not None:
            u, m = replace
            x = x - (x @ u)[..., None] * u + m[k].view(*([1] * (x.dim() - 2)), -1, 1) * u if m[k].dim() else \
                x - (x @ u)[..., None] * u + m[k] * u
        if frozen is not None:
            s, s0 = frozen[k]
            c = x - x.mean(-1, keepdim=True)
            inv = torch.full(c.shape[:-1] + (1,), s, dtype=c.dtype)
            inv[:, 0] = s0
            out = g * c * inv + b
        else:
            out = ln_std(x, g, b)
        k += 1
        return out

    for bp in P["blocks"]:
        x = x + attn(LNk(x, bp["g1"], bp["b1"]) @ bp["Wqkv"].T + bp["bqkv"], bp)
        x = x + F.gelu(LNk(x, bp["g2"], bp["b2"]) @ bp["W1"].T + bp["bf1"]) @ bp["W2"].T + bp["bf2"]
    return LNk(x, P["gf"], P["bf"]) @ P["Wh"].T + P["bh"]


def fold_C(P):
    d = P["E"].shape[1]
    C = torch.eye(d, dtype=torch.double) - 1.0 / d
    Q = dict(P)
    Q["E"], Q["P"] = P["E"] @ C, P["P"] @ C
    Q["blocks"] = []
    for bp in P["blocks"]:
        bq = dict(bp)
        bq["Wo"], bq["bo"] = C @ bp["Wo"], C @ bp["bo"]
        bq["W2"], bq["bf2"] = C @ bp["W2"], C @ bp["bf2"]
        Q["blocks"].append(bq)
    return Q


def explicit(P, u, m):
    """The smaller network on y = V^T x, with pre-division biases and m_k^2 in the norm."""
    d = P["E"].shape[1]
    # orthonormal basis of the complement of u
    Qm, _ = torch.linalg.qr(torch.cat([u[:, None], torch.eye(d, dtype=torch.double)], 1))
    V = Qm[:, 1:d]

    def site(W, bias, g, b, k):
        return {"A": W @ torch.diag(g) @ V, "c": m[k][..., None] * (W @ (g * u)), "e": W @ b + bias, "m2": m[k] ** 2}

    S = {"E": P["E"] @ V, "P": P["P"] @ V, "sites": [], "blocks": []}
    k = 0
    for bp in P["blocks"]:
        s1 = site(bp["Wqkv"], bp["bqkv"], bp["g1"], bp["b1"], k); k += 1
        s2 = site(bp["W1"], bp["bf1"], bp["g2"], bp["b2"], k); k += 1
        S["blocks"].append({"s1": s1, "s2": s2, "Wo": V.T @ bp["Wo"], "bo": V.T @ bp["bo"],
                            "W2": V.T @ bp["W2"], "bf2": V.T @ bp["bf2"]})
    S["sf"] = site(P["Wh"], P["bh"], P["gf"], P["bf"], k)
    return S


def forward_explicit(S, idx, d):
    y = S["E"][idx] + S["P"][: idx.shape[1]]

    def apply(y, st):
        m2 = st["m2"]
        m2 = m2.view(1, -1, 1) if m2.dim() else m2
        s = torch.sqrt((y.pow(2).sum(-1, keepdim=True) + m2) / d + EPS)
        c = st["c"]
        c = c.unsqueeze(0) if c.dim() == 2 else c
        return (y @ st["A"].T + c) / s + st["e"]

    for bp in S["blocks"]:
        h = apply(y, bp["s1"])
        B, T, _ = h.shape
        D, H = LF.D, LF.H
        q, k, v = h.view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        y = y + a @ bp["Wo"].T + bp["bo"]
        y = y + F.gelu(apply(y, bp["s2"])) @ bp["W2"].T + bp["bf2"]
    return apply(y, S["sf"])


def ce(z, Y):
    return float(F.cross_entropy(z.flatten(0, 1).float(), Y.flatten()))


@torch.no_grad()
def main():
    tr, va, V = LF.corpus()
    model = LF.GPT(V)
    model.load_state_dict(torch.load(os.path.join(OUT, "gpt_char.pt")))
    model.eval()
    g = torch.Generator().manual_seed(1)
    Xv, Yv = LF.batches(va, 512, g)
    Xt, _ = LF.batches(tr, 512, torch.Generator().manual_seed(2))
    P = params(model)
    d = P["E"].shape[1]
    z0 = forward_ref(P, Xv)
    base = ce(z0, Yv)
    err_model = float((z0.float() - model(Xv)).abs().max())
    Q = fold_C(P)
    zq = forward_ref(Q, Xv)
    err_fold = float((zq - z0).abs().max())
    print(f"base loss {base:.4f}; functional forward vs module: {err_model:.1e}; step A (fold C): {err_fold:.1e}",
          flush=True)

    # candidate directions: pooled covariance of the 9 LayerNorm inputs (training tokens), within 1-perp
    rec = []
    forward_ref(Q, Xt, record=rec)
    one = torch.ones(d, dtype=torch.double) / d ** 0.5
    Pperp = torch.eye(d, dtype=torch.double) - torch.outer(one, one)
    Cpool = 0
    for x in rec:
        a = x.reshape(-1, d)
        a = a - a.mean(0)
        C = a.T @ a / len(a)
        Cpool = Cpool + C / torch.trace(C)
    Cpool = Pperp @ Cpool @ Pperp + 10 * torch.outer(one, one)        # push the 1-direction to the top
    _, E = torch.linalg.eigh(Cpool)

    def means(u, Xs, per_pos):
        """training means of u^T x at the 9 LayerNorm inputs, computed depth by depth with
        the earlier replacements already in place."""
        m = []
        for k in range(9):
            rec = []
            forward_ref(Q, Xs, replace=(u, m + [torch.tensor(0.0, dtype=torch.double)] * (9 - len(m))) if m else None,
                        record=rec)
            v = rec[k] @ u
            m.append(v.mean(0) if per_pos else v.mean())
        return m

    # frozen-sigma constants of the plain model (for the comparison)
    rec = []
    forward_ref(Q, Xt, record=rec)
    frozen = []
    for x in rec:
        c = x - x.mean(-1, keepdim=True)
        sig = torch.sqrt(c.pow(2).mean(-1) + EPS)
        frozen.append((float(1 / sig[:, 1:].mean()), float(1 / sig[:, 0].mean())))
    loss_frozen_only = ce(forward_ref(Q, Xv, frozen=frozen), Yv)

    rows = []
    for i in range(8):
        u = E[:, i]
        for per_pos in (False, True):
            m = means(u, Xt, per_pos)
            zr = forward_ref(Q, Xv, replace=(u, m))
            row = dict(direction=i, per_position=per_pos, loss=ce(zr, Yv))
            if not per_pos:
                S = explicit(Q, u, m)
                row["explicit_vs_replaced"] = float((forward_explicit(S, Xv, d) - zr).abs().max())
                row["loss_with_frozen_sigma"] = ce(forward_ref(Q, Xv, replace=(u, m), frozen=frozen), Yv)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["base_loss"] = base
    df["rel"] = (df.loss - base) / base
    df.to_csv(os.path.join(OUT, "exact_route.csv"), index=False)
    pd.set_option("display.width", 160)
    print(df.round(6).to_string(index=False), flush=True)
    print(f"freezing sigma at all 9 LayerNorms, no stream move: loss {loss_frozen_only:.4f} "
          f"({100 * (loss_frozen_only - base) / base:+.2f}%)", flush=True)

    # several directions at once (exact route, position-independent means)
    seq = []
    for r in (1, 2, 4, 8, 16, 32):
        U = E[:, :r]
        Pi = torch.eye(d, dtype=torch.double) - U @ U.T
        m = []
        for k in range(9):                 # vector means of U^T x, depth by depth
            rec = []

            def fwd(Xs, mm):
                return forward_ref_multi(Q, Xs, U, Pi, mm, rec)
            rec.clear()
            fwd(Xt, m + [None] * (9 - len(m)))
            m.append((rec[k] @ U).reshape(-1, r).mean(0))
        l = ce(forward_ref_multi(Q, Xv, U, Pi, m, None), Yv)
        seq.append(dict(r=r, loss=l, rel=(l - base) / base, freed=6576 * r - 3764))
        print(f"r = {r:2d} directions removed: loss {l:.4f} ({100 * (l - base) / base:+.2f}%), "
              f"parameters freed {6576 * r - 3764}", flush=True)
    pd.DataFrame(seq).to_csv(os.path.join(OUT, "exact_route_multi.csv"), index=False)
    print("EXACT DONE", flush=True)


def forward_ref_multi(P, idx, U, Pi, m, record):
    """Replacement of r directions at once: x -> Pi x + U m_k at each LayerNorm input."""
    x = P["E"][idx] + P["P"][: idx.shape[1]]
    k = 0

    def LNk(x, g, b):
        nonlocal k
        if record is not None:
            record.append(x)
        if m[k] is not None:
            x = x @ Pi + U @ m[k]
        k += 1
        return ln_std(x, g, b)

    for bp in P["blocks"]:
        x = x + attn(LNk(x, bp["g1"], bp["b1"]) @ bp["Wqkv"].T + bp["bqkv"], bp)
        x = x + F.gelu(LNk(x, bp["g2"], bp["b2"]) @ bp["W1"].T + bp["bf1"]) @ bp["W2"].T + bp["bf2"]
    return LNk(x, P["gf"], P["bf"]) @ P["Wh"].T + P["bh"]


if __name__ == "__main__":
    main()
