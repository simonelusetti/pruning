"""
Do trained ResNets have removable stream channels?

Network (post-activation ResNet, BN folded after training):
    x0 = maxpool(ReLU(conv_stem(u)))                      16 ch, 14x14   stage 1
    x1 = ReLU(x0 + B0 ReLU(A0 x0))                        16 ch, 14x14
    x2 = ReLU(x1 + B1 ReLU(A1 x1))                        16 ch, 14x14
    x3 = ReLU(S2 x2 + B2 ReLU(A2 x2))                     32 ch, 7x7     stage 2 (A2, S2 stride 2)
    x4 = ReLU(x3 + B3 ReLU(A3 x3))                        32 ch, 7x7
    z  = Wf mean(x4) + bf
A_k, B_k are 3x3 convs, S2 is 1x1 stride 2; all carry the folded BN bias.

Moves measured, each a single move on the trained network (no path):
  stream channel j of stage 1 (used by A0 at x0, A1 at x1, A2 and S2 at x2) or of stage 2
  (used by A3 at x3, fc at x4).  At every place it is used, its contribution
  W[o, j] * x_j is replaced by the least-squares fit on
      zero  nothing (no compensation)
      mean  a constant                         -> folded into that reader's bias
      pw    x_i at the same pixel + constant   -> folded into W[o, i] (centre tap) and bias
      sp    3x3 neighbourhood of x_i + const   -> folded into W[o, i] (whole kernel) and bias
  then every weight that computes channel j (stem/S2 row j, B_k rows j) is deleted.
  The partner i is the one with the smallest total squared fit error (no loss used).
  inner channel j of block k (computed by A_k, used only by B_k): the same four moves
  (her unit moves, for reference).
  block_mean: B_k ReLU(A_k x) replaced by its per-pixel training mean (her head_mean).

Fits on 2 000 training images; loss change on valB (8 000 held-out images), paired per
sample.  'pass' = her stop rule with rho = 1 %:  mean dL <= 2 SE + 0.01 * base loss.

Usage (each command stops after budget_s seconds and resumes on the next call):
  python resnet_stream_test.py train   [budget_s] [seeds...]   repeat until every seed says 'trained'
  python resnet_stream_test.py measure [budget_s] [seeds...]   repeat until every seed says 'measured'
  python resnet_stream_test.py summary 0 [seeds...]
"""
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn_fm import load_fashion  # noqa: E402

torch.set_num_threads(4)
OUT = os.environ.get("RESNET_RESULTS", os.path.join(HERE, "results_resnet"))
EPOCHS, LR, BATCH = 10, 3e-3, 128


# ---------------------------------------------------------------- model + training
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.n2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc, self.ns = nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm2d(cout)

    def forward(self, x):
        f = self.n2(self.c2(F.relu(self.n1(self.c1(x)))))
        return F.relu((x if self.sc is None else self.ns(self.sc(x))) + f)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem, self.nstem = nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
        self.blocks = nn.ModuleList([Block(16, 16, 1), Block(16, 16, 1), Block(16, 32, 2), Block(32, 32, 1)])
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.nstem(self.stem(x))), 2)
        for b in self.blocks:
            x = b(x)
        return self.fc(x.mean((2, 3)))


def train(seed, budget, fit):
    os.makedirs(OUT, exist_ok=True)
    final, ckpt = os.path.join(OUT, f"resnet_seed{seed}.pt"), os.path.join(OUT, f"train_ckpt_seed{seed}.pt")
    if os.path.exists(final):
        return True
    t0, ok = time.time(), True
    torch.manual_seed(seed)
    net = ResNet()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ep0 = 0
    if os.path.exists(ckpt):
        C = torch.load(ckpt, weights_only=False)
        net.load_state_dict(C["net"]); opt.load_state_dict(C["opt"]); ep0 = C["epoch"]
    X, y = fit
    for ep in range(ep0, EPOCHS):
        if ep > ep0 and time.time() - t0 > budget:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ckpt)
            print(f"epoch {ep}/{EPOCHS}", flush=True)
            return False
        for g in opt.param_groups:
            g["lr"] = LR if ep < EPOCHS - 2 else LR / 10
        net.train()
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(1000 * seed + ep))
        for b in range(0, len(y), BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            F.cross_entropy(net(X[idx]), y[idx]).backward()
            opt.step()
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, ckpt)
    torch.save(net.state_dict(), final)
    try:
        os.remove(ckpt)
    except PermissionError:
        os.replace(ckpt, ckpt + ".finished")
    return True


# ---------------------------------------------------------------- folded network
def _fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return [(conv.weight * s[:, None, None, None]).detach().clone(), (bn.bias - bn.running_mean * s).detach().clone()]


def fold(net):
    P = {"stem": _fold(net.stem, net.nstem)}
    for k, b in enumerate(net.blocks):
        P[f"A{k}"], P[f"B{k}"] = _fold(b.c1, b.n1), _fold(b.c2, b.n2)
        if b.sc is not None:
            P[f"S{k}"] = _fold(b.sc, b.ns)
    P["fc"] = [net.fc.weight.detach().clone(), net.fc.bias.detach().clone()]
    return P


def clone(P):
    return {k: [t.clone() for t in v] for k, v in P.items()}


@torch.no_grad()
def forward(P, X, keep=False, block_mean=None):
    x = F.max_pool2d(F.relu(F.conv2d(X, *P["stem"], padding=1)), 2)
    xs, hs = [x], []
    for k in range(4):
        st = 2 if k == 2 else 1
        h = F.relu(F.conv2d(x, *P[f"A{k}"], stride=st, padding=1))
        f = F.conv2d(h, *P[f"B{k}"], padding=1)
        if block_mean is not None and k in block_mean:
            f = block_mean[k].expand_as(f)
        x = F.relu((F.conv2d(x, *P["S2"], stride=2) if k == 2 else x) + f)
        xs.append(x); hs.append(h)
    z = F.linear(x.mean((2, 3)), *P["fc"])
    return (z, xs, hs) if keep else z


@torch.no_grad()
def forward_parts(P, X):
    """Like forward, but also returns each block's branch output f_k (for block_mean)."""
    x = F.max_pool2d(F.relu(F.conv2d(X, *P["stem"], padding=1)), 2)
    fs = []
    for k in range(4):
        st = 2 if k == 2 else 1
        f = F.conv2d(F.relu(F.conv2d(x, *P[f"A{k}"], stride=st, padding=1)), *P[f"B{k}"], padding=1)
        x = F.relu((F.conv2d(x, *P["S2"], stride=2) if k == 2 else x) + f)
        fs.append(f)
    return fs


def nnz(P):
    return sum(int((t != 0).sum()) for v in P.values() for t in v)


# ---------------------------------------------------------------- readers and Gram matrices
# (key, which activation it reads, kernel, stride, pad); 'fc' reads the spatial mean
STREAM = {1: {"readers": [("A0", ("x", 0), 3, 1, 1), ("A1", ("x", 1), 3, 1, 1),
                          ("A2", ("x", 2), 3, 2, 1), ("S2", ("x", 2), 1, 2, 0)],
              "writers": ["stem", "B0", "B1"], "C": 16},
          2: {"readers": [("A3", ("x", 3), 3, 1, 1), ("fc", ("x", 4), 0, 0, 0)],
              "writers": ["S2", "B2", "B3"], "C": 32}}
INNER = {k: {"readers": [(f"B{k}", ("h", k), 3, 1, 1)], "writers": [f"A{k}"], "C": 32 if k >= 2 else 16}
         for k in range(4)}


@torch.no_grad()
def gram(P, X, src, ks, st, pad, chunk=200):
    """Second-moment matrix of [patch features of all channels, 1] over positions and images."""
    G = None
    for b in range(0, len(X), chunk):
        _, xs, hs = forward(P, X[b:b + chunk], keep=True)
        a = (xs if src[0] == "x" else hs)[src[1]]
        if ks == 0:
            Phi = a.mean((2, 3))
        else:
            Phi = F.unfold(a, ks, padding=pad, stride=st).transpose(1, 2).reshape(-1, a.shape[1] * ks * ks)
        Phi = torch.cat([Phi, torch.ones(len(Phi), 1)], 1).double()
        G = Phi.T @ Phi if G is None else G + Phi.T @ Phi
    return G


def wcols(P, key, j, ks):
    W = P[key][0]
    return W[:, j].reshape(W.shape[0], -1).double() if ks else W[:, j:j + 1].double()


def fit(G, Wj, j, i, ks, kind):
    """Least-squares fit of the contribution Wj * x_j on the chosen features of x_i.
    Returns (theta (|S| x O), residual SSE, total SSE)."""
    kk = max(ks, 1) ** 2
    J = list(range(j * kk, (j + 1) * kk))
    const = G.shape[0] - 1
    if kind in ("zero", "mean"):
        S = [const]
    elif kind == "pw" or ks <= 1:
        S = [i * kk + kk // 2, const]
    else:
        S = list(range(i * kk, (i + 1) * kk)) + [const]
    sst = float((Wj @ G[J][:, J] * Wj).sum())
    if kind == "zero":
        return None, sst, sst
    Gss, H = G[S][:, S], G[S][:, J] @ Wj.T
    th = torch.linalg.solve(Gss + 1e-9 * torch.eye(len(S), dtype=Gss.dtype) * Gss.diagonal().mean(), H)
    return th, sst - float((th * H).sum()), sst


def apply_fit(P, key, j, i, ks, kind, th):
    W, b = P[key]
    if kind != "zero":
        b += th[-1].float()
        if kind != "mean":
            if ks == 0:
                W[:, i] += th[0].float()
            elif kind == "pw" or ks == 1:
                c = ks // 2
                W[:, i, c, c] += th[0].float()
            else:
                W[:, i] += th[:-1].T.float().reshape(W.shape[0], ks, ks)
    W[:, j] = 0


def kill_writer(P, key, j):
    P[key][0][j] = 0
    P[key][1][j] = 0


# ---------------------------------------------------------------- measurement
def per_sample(P, X, y, **kw):
    return F.cross_entropy(forward(P, X, **kw), y, reduction="none")


def measure(seed, deadline):
    """Resumable: one CSV per group under OUT/stream_units, rows appended unit by unit."""
    udir = os.path.join(OUT, "stream_units")
    os.makedirs(udir, exist_ok=True)
    data, _ = load_fashion(seed, n_valA=2000, n_prune_train=2000)
    net = ResNet()
    net.load_state_dict(torch.load(os.path.join(OUT, f"resnet_seed{seed}.pt"), weights_only=False))
    net.eval()
    P0 = fold(net)
    Xt = data["train"][0]
    XB, yB = data["valB"]
    Xte, yte = data["test"]
    assert torch.allclose(net(XB[:500]), forward(P0, XB[:500]), atol=1e-4), "BN fold is not exact"
    L0 = per_sample(P0, XB, yB)
    base = float(L0.mean())
    k0 = nnz(P0)
    info = os.path.join(udir, f"seed{seed}__base.txt")
    if not os.path.exists(info):
        acc = float((forward(P0, Xte).argmax(1) == yte).float().mean())
        with open(info, "w") as fh:
            fh.write(f"base_valB_loss {base:.6f}\ntest_acc {acc:.6f}\nparams {k0}\n")
        print(f"[seed {seed}] base valB loss {base:.4f}  test acc {acc:.4f}  params {k0}", flush=True)

    def score(P, **kw):
        d = per_sample(P, XB, yB, **kw) - L0
        m, se = float(d.mean()), float(d.std() / len(d) ** 0.5)
        return m, se, bool(m <= 2 * se + 0.01 * base)

    def save(f, rows):
        pd.DataFrame(rows).assign(base_loss=base).to_csv(f, mode="a", header=not os.path.exists(f), index=False)

    groups = [(f"stream{s}", g) for s, g in STREAM.items()] + [(f"inner{k}", g) for k, g in INNER.items()]
    for gname, g in groups:
        f = os.path.join(udir, f"seed{seed}__{gname}.csv")
        done = set(pd.read_csv(f)["unit"]) if os.path.exists(f) else set()
        C = g["C"]
        if len(done) == C:
            continue
        Gs = {r[0]: gram(P0, Xt, r[1], r[2], r[3], r[4]) for r in g["readers"]}
        for j in range(C):
            if j in done:
                continue
            if time.time() > deadline:
                return False
            rows = []
            for kind in ("zero", "mean", "pw", "sp"):
                partners = [None] if kind in ("zero", "mean") else [i for i in range(C) if i != j]
                best = None
                for i in partners:
                    fits, sse, sst = [], 0.0, 0.0
                    for key, _, ks, _, _ in g["readers"]:
                        th, e, t = fit(Gs[key], wcols(P0, key, j, ks), j, i, ks, kind)
                        fits.append((key, ks, th)); sse += e; sst += t
                    if best is None or sse < best[1]:
                        best = (i, sse, sst, fits)
                i, sse, sst, fits = best
                P = clone(P0)
                for key, ks, th in fits:
                    apply_fit(P, key, j, i, ks, kind, th)
                for key in g["writers"]:
                    kill_writer(P, key, j)
                m, se, ok = score(P)
                rows.append(dict(seed=seed, group=gname, unit=j, kind=kind, partner=i,
                                 unexplained=sse / max(sst, 1e-30), dloss=m, se=se, passes=ok,
                                 freed=k0 - nnz(P)))
            save(f, rows)
        print(f"[seed {seed}] {gname} done", flush=True)

    f = os.path.join(udir, f"seed{seed}__block.csv")
    if not os.path.exists(f):
        fs = None
        for b in range(0, len(Xt), 250):
            part = forward_parts(P0, Xt[b:b + 250])
            fs = [x.sum(0) for x in part] if fs is None else [a + x.sum(0) for a, x in zip(fs, part)]
        means = [x / len(Xt) for x in fs]
        rows = []
        for k in range(4):
            m, se, ok = score(P0, block_mean={k: means[k]})
            freed = sum(int((t != 0).sum()) for t in P0[f"A{k}"] + P0[f"B{k}"]) - means[k].numel()
            rows.append(dict(seed=seed, group="block", unit=k, kind="block_mean", partner=None,
                             unexplained=float("nan"), dloss=m, se=se, passes=ok, freed=freed))
        save(f, rows)
    return True


def load_all(seeds):
    udir = os.path.join(OUT, "stream_units")
    return pd.concat([pd.read_csv(os.path.join(udir, n)) for n in sorted(os.listdir(udir))
                      if n.endswith(".csv") and int(n.split("__")[0][4:]) in seeds], ignore_index=True)


def summary(df):
    d = df.copy()
    d["group"] = d["group"].str.replace(r"inner\d", "inner", regex=True)
    tab = d.groupby(["group", "kind"]).agg(n=("dloss", "size"), median_dloss=("dloss", "median"),
                                           pass_frac=("passes", "mean"),
                                           median_unexpl=("unexplained", "median"),
                                           median_freed=("freed", "median"))
    return tab


if __name__ == "__main__":
    mode = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 150
    seeds = [int(a) for a in sys.argv[3:]] or [0]
    t0, ok = time.time(), True
    for seed in seeds:
        if mode == "train":
            _, fit_data = load_fashion(seed, n_valA=2000, n_prune_train=2000)
            ok = train(seed, budget - (time.time() - t0), fit_data)
            print(f"[seed {seed}] " + ("trained" if ok else "not yet"), flush=True)
        elif mode == "measure":
            ok = measure(seed, t0 + budget)
            print(f"[seed {seed}] " + ("measured" if ok else "not yet"), flush=True)
        if not ok:
            break
    if mode == "summary":
        pd.set_option("display.width", 160)
        print(summary(load_all(seeds)).to_string(float_format=lambda v: f"{v:.4g}"))
