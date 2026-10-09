"""
Can one shared subspace describe the stream x at every depth of her transformer?

Model: Veronica's cached base model (NNpruning/results/transformer/base_state_seed0.pt,
1 layer, d = 64, a + b mod 59, the network of her paper), on the same data split.

Stream vectors and where they are used (positions t = 0, 1, 2; only t = 2 reaches the loss):
  x0_t  used by W_Q (t = 2), W_K and W_V (all t)
  x1_2  used by W_in
  x2_2  used by W_U
For each depth k the covariance C_k is taken after subtracting the per-position mean
(x0 over t = 0, 1; x0_2 is the same for every input).

The move with r directions removed: at every use, x is replaced by
    Pi x + (I - Pi) m_t^k,        Pi = V V^T,  V in R^{64 x (64-r)} orthonormal,
m_t^k = training mean of x at depth k, position t (computed after the earlier depths
were already replaced).  With one V for all depths this is exactly the folded network
(every matrix applied to x becomes M V, every added term a becomes V^T a, constants go
into biases; checked numerically).  With a different V_k per depth it is only an upper
bound: Pi_1 x1 = Pi_1 x0 + ..., and Pi_1 x0 uses components of x0 that depth 0 discarded,
so a folded version would need more than a (64-r)^2 matrix between depths.

Methods compared, r = 0 .. 63:
  shared-pca     one V: smallest eigen-directions of C_0/tr C_0 + C_1/tr C_1 + C_2/tr C_2
  shared-greedy  one V: directions removed one at a time; candidates = the 3 smallest
                 eigenvectors (within the kept subspace) of the pooled matrix and of each
                 C_k; the one with the lowest training loss is removed
  per-depth-pca  upper bound: V_k from the smallest eigen-directions of C_k alone
Parameters freed by the shared move: 891 per direction (one column/row of W_E, W_pos,
W_Q, W_K, W_V, W_O, W_in, W_out, b_out, W_U) minus 448 new bias numbers for q, k, v
(the constants (I - Pi) m folded after W_Q, W_K, W_V, which have no bias).

Usage:  python stream_directions.py            (writes results_stream/*.csv)
"""
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "NNpruning"))
from nnprune import transformer as T  # noqa: E402

torch.set_num_threads(4)
OUT = os.environ.get("STREAM_RESULTS", os.path.join(HERE, "results_stream"))
D, PER_DIR, NEW_BIAS = 64, 891, 448


def setup():
    data = T.make_modadd_data(p=59, train_frac=0.5, seed=0)
    ad = T.TransformerAdapter(data, p=59)
    s0 = torch.load(os.path.join(ROOT, "NNpruning", "results", "transformer", "base_state_seed0.pt"),
                    weights_only=False)
    return ad, s0, data


class Evaluator:
    """Means m_t^k are training means.  They are computed on the training set with the
    same projections, then frozen and reused on validation / test."""

    def __init__(self, ad, s0, data):
        self.ad, self.S, self.P = ad, s0.struct, ad.effective(s0)
        self.data = data

    @torch.no_grad()
    def run(self, proj, split):
        Xtr = self.data["train"][0]
        means = None
        if proj is not None:                       # training means, depth by depth
            means = []
            x0 = self.P["W_E"][Xtr] + self.P["W_pos"]
            means.append(x0.mean(0))
            _, (_, x1, _) = forward_fixed(self.ad, self.P, self.S, Xtr, proj, means + [None, None], upto=1)
            means.append(x1.mean(0))
            _, (_, _, x2) = forward_fixed(self.ad, self.P, self.S, Xtr, proj, means + [None], upto=2)
            means.append(x2.mean(0))
        X, y = self.data[split]
        z, _ = forward_fixed(self.ad, self.P, self.S, X, proj, means)
        return float(F.cross_entropy(z, y)), float((z.argmax(1) == y).float().mean())


@torch.no_grad()
def forward_fixed(ad, P, S, X, proj, means, upto=3):
    """Same as forward, with given (frozen) means; depths >= upto are left unreplaced."""
    def use(x, k):
        if proj is None or k >= upto or means[k] is None:
            return x
        return x @ proj[k] + means[k] @ (torch.eye(D) - proj[k])

    x0 = P["W_E"][X] + P["W_pos"]
    u0 = use(x0, 0)
    q = torch.einsum("btd,hde->bhte", u0, P["W_Q"])
    k = torch.einsum("btd,hde->bhte", u0, P["W_K"])
    v = torch.einsum("btd,hde->bhte", u0, P["W_V"])
    sc = torch.einsum("bhte,bhse->bhts", q, k) / (ad.dh ** 0.5)
    sc = sc.masked_fill(~ad.causal, float("-inf"))
    attn = torch.softmax(sc, -1)
    attn = torch.where(S["head_fixed"][:, None, None], S["A_fixed"], attn)
    ho = torch.einsum("bhte,hed->bhtd", torch.einsum("bhts,bhse->bhte", attn, v), P["W_O"])
    ho = ho * S["head_alive"].to(ho.dtype)[:, None, None]
    x1 = x0 + ho.sum(-3) + P["B_attn"]
    u1 = use(x1, 1)
    act = torch.relu(u1 @ P["W_in"].T + P["b_in"])
    act = act * (S["unit_type"] == T.ACTIVE).to(act.dtype)
    x2 = x1 + act @ P["W_out"].T + P["b_out"]
    u2 = use(x2, 2)
    return u2[:, -1] @ P["W_U"] + P["b_U"], (x0, x1, x2)


def covariances(ev):
    X = ev.data["train"][0]
    _, (x0, x1, x2) = forward_fixed(ev.ad, ev.P, ev.S, X, None, None)
    a0 = torch.cat([x0[:, 0] - x0[:, 0].mean(0), x0[:, 1] - x0[:, 1].mean(0)]).double()
    a1 = (x1[:, 2] - x1[:, 2].mean(0)).double()
    a2 = (x2[:, 2] - x2[:, 2].mean(0)).double()
    return [a.T @ a / len(a) for a in (a0, a1, a2)], float((x0[:, 2] - x0[:, 2].mean(0)).abs().max())


def basis_drop(C, r):
    """Orthonormal basis of the span of the 64 - r largest eigen-directions of C."""
    w, U = torch.linalg.eigh(C)
    return U[:, r:].float()


def proj_of(V):
    return V @ V.T


def main():
    os.makedirs(OUT, exist_ok=True)
    ad, s0, data = setup()
    ev = Evaluator(ad, s0, data)
    Cs, const2 = covariances(ev)
    base = {sp: ev.run(None, sp) for sp in ("train", "val", "test")}
    print(f"base  train loss {base['train'][0]:.2e}  val loss {base['val'][0]:.2e}  "
          f"test acc {base['test'][1]:.3f};  x0 at t=2 varies by at most {const2:.1e}", flush=True)

    # spectra, and how much of each depth's variance a shared basis keeps
    pooled = sum(C / torch.trace(C) for C in Cs)
    rows = []
    for k, C in enumerate(Cs):
        w = torch.linalg.eigvalsh(C).flip(0)
        for i, lam in enumerate(w.tolist()):
            rows.append(dict(depth=k, index=i, eig=lam, eig_rel=lam / float(w[0])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "spectra.csv"), index=False)

    var_rows = []
    for r in range(D):
        Vs = basis_drop(pooled, r).double()
        for k, C in enumerate(Cs):
            tot = float(torch.trace(C))
            own = float(torch.linalg.eigvalsh(C).flip(0)[: D - r].sum())
            shared = float(torch.trace(Vs.T @ C @ Vs))
            var_rows.append(dict(r=r, depth=k, kept_own=own / tot, kept_shared=shared / tot))
    pd.DataFrame(var_rows).to_csv(os.path.join(OUT, "variance_kept.csv"), index=False)

    # functional sweep
    out = []

    def record(method, r, projs):
        tr = ev.run(projs, "train"); va = ev.run(projs, "val"); te = ev.run(projs, "test")
        out.append(dict(method=method, r=r, train_loss=tr[0], val_loss=va[0], val_acc=va[1],
                        test_loss=te[0], test_acc=te[1],
                        freed=PER_DIR * r - (NEW_BIAS if r else 0)))

    for r in range(D):
        V = basis_drop(pooled, r)
        record("shared-pca", r, [proj_of(V)] * 3)
        record("per-depth-pca", r, [proj_of(basis_drop(C, r)) for C in Cs])
    print("pca sweeps done", flush=True)

    V = torch.eye(D)
    record("shared-greedy", 0, [proj_of(V)] * 3)
    for r in range(1, D):
        cands = []
        for M in [pooled] + [C / torch.trace(C) for C in Cs]:
            Mr = V.double().T @ M @ V.double()
            _, E = torch.linalg.eigh(Mr)
            for i in range(min(3, E.shape[1])):
                cands.append(E[:, i].float())
        best = None
        for e in cands:
            Qr, _ = torch.linalg.qr(torch.cat([e[:, None], torch.eye(V.shape[1])], 1))
            Vn = V @ Qr[:, 1:V.shape[1]]
            loss = ev.run([proj_of(Vn)] * 3, "train")[0]
            if best is None or loss < best[0]:
                best = (loss, Vn)
        V = best[1]
        record("shared-greedy", r, [proj_of(V)] * 3)
    print("greedy done", flush=True)

    df = pd.DataFrame(out)
    df["base_val_loss"] = base["val"][0]
    df.to_csv(os.path.join(OUT, "sweep.csv"), index=False)
    return df


if __name__ == "__main__":
    df = main()
    pd.set_option("display.width", 160)
    sel = df[df.r.isin([0, 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56])]
    print(sel.pivot(index="r", columns="method", values="val_loss").to_string(float_format=lambda v: f"{v:.3g}"))
    print(sel.pivot(index="r", columns="method", values="test_acc").to_string(float_format=lambda v: f"{v:.3f}"))
