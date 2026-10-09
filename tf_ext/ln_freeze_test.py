"""
How constant is sigma(x) at the LayerNorms of a trained multi-layer transformer, and
what does freezing it cost?

Model: pre-LN decoder transformer (GPT-2 layout), 4 blocks, d = 128, 4 heads, MLP 512,
context 64, learned positions, final LayerNorm, trained from scratch as a character
model on the Python standard-library sources (top-level .py files of python3.13; no
download needed).  Each block:  x <- x + Attn(LN1(x));  x <- x + MLP(LN2(x)).

For every LayerNorm (ln1, ln2 of each block, final ln):
  spread   coefficient of variation of sigma(x) over held-out tokens, all positions and
           without position 0; ratio of mean sigma at position 0 to the rest
  freeze   1/sigma(x) -> s (one constant per LayerNorm, s = 1 / mean sigma, as in
           Heimersheim 2024) and also with a separate constant for position 0;
           loss change for that LayerNorm alone and for all LayerNorms at once.

Usage:  python ln_freeze_test.py            (writes results_ln/ln_stats.csv, ln_freeze.csv)
"""
import glob
import math
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("LN_RESULTS", os.path.join(HERE, "results_ln"))
D, H, L, CTX, MLP = 128, 4, 4, 64, 512
STEPS, BATCH, LR = int(os.environ.get("STEPS", "4000")), 64, 3e-3


def corpus():
    # corpus: top-level .py files of a Python standard library (the results in results_ln used
    # /usr/lib/python3.13); override with CORPUS_GLOB on another machine
    files = sorted(glob.glob(os.environ.get("CORPUS_GLOB", "/usr/lib/python3.13/*.py")))
    if not files:
        raise FileNotFoundError("no corpus files: set CORPUS_GLOB, e.g. '/usr/lib/python3.12/*.py'")
    text = "".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)
    chars = sorted(set(text))
    table = {c: i for i, c in enumerate(chars)}
    idx = torch.tensor([table[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(idx))
    return idx[:n], idx[n:], len(chars)


class LN(nn.Module):
    """LayerNorm with an optional frozen scale: 1/sigma(x) -> s (s0 at position 0)."""

    def __init__(self, d):
        super().__init__()
        self.g, self.b = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.frozen = None            # None or (s, s0)
        self.record = None            # list to append sigma to

    def forward(self, x):
        c = x - x.mean(-1, keepdim=True)
        sig = torch.sqrt(c.pow(2).mean(-1, keepdim=True) + 1e-5)
        if self.record is not None:
            self.record.append(sig.squeeze(-1).detach())
        if self.frozen is None:
            inv = 1 / sig
        else:
            s, s0 = self.frozen
            inv = torch.full_like(sig, s)
            inv[:, 0] = s0
        return self.g * c * inv + self.b


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = LN(D), LN(D)
        self.qkv, self.o = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, MLP), nn.Linear(MLP, D)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Parameter(torch.randn(CTX, D) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.lnf, self.head = LN(D), nn.Linear(D, V)

    def lns(self):
        out = []
        for i, b in enumerate(self.blocks):
            out += [(f"block{i}.ln1", b.ln1), (f"block{i}.ln2", b.ln2)]
        return out + [("final", self.lnf)]

    def forward(self, idx):
        x = self.emb(idx) + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def batches(data, n, g):
    i = torch.randint(0, len(data) - CTX - 1, (n,), generator=g)
    X = torch.stack([data[j:j + CTX] for j in i])
    Y = torch.stack([data[j + 1:j + CTX + 1] for j in i])
    return X, Y


@torch.no_grad()
def loss_on(model, X, Y):
    return float(F.cross_entropy(model(X).flatten(0, 1), Y.flatten()))


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, va, V = corpus()
    print(f"corpus: {len(tr) + len(va)} characters, vocabulary {V}", flush=True)
    torch.manual_seed(0)
    model = GPT(V)
    ck = os.path.join(OUT, "gpt_char.pt")
    if os.path.exists(ck):
        model.load_state_dict(torch.load(ck))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
        g = torch.Generator().manual_seed(0)
        t0 = time.time()
        for step in range(STEPS):
            for gr in opt.param_groups:
                gr["lr"] = LR * min(1, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
            X, Y = batches(tr, BATCH, g)
            loss = F.cross_entropy(model(X).flatten(0, 1), Y.flatten())
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), ck)
    model.eval()
    g = torch.Generator().manual_seed(1)
    Xv, Yv = batches(va, 512, g)
    base = loss_on(model, Xv, Yv)
    print(f"held-out loss {base:.4f} nats/char", flush=True)

    # sigma statistics
    for _, ln in model.lns():
        ln.record = []
    with torch.no_grad():
        model(Xv)
    rows, consts = [], {}
    for name, ln in model.lns():
        sig = torch.cat(ln.record)                       # (n, T)
        ln.record = None
        rest = sig[:, 1:]
        consts[name] = (float(1 / rest.mean()), float(1 / sig[:, 0].mean()), float(1 / sig.mean()))
        rows.append(dict(ln=name, mean_sigma=float(sig.mean()),
                         cv_all=float(sig.std() / sig.mean()), cv_without_pos0=float(rest.std() / rest.mean()),
                         pos0_over_rest=float(sig[:, 0].mean() / rest.mean()),
                         p01=float(rest.quantile(0.01) / rest.mean()), p99=float(rest.quantile(0.99) / rest.mean())))
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(OUT, "ln_stats.csv"), index=False)
    print(stats.round(3).to_string(index=False), flush=True)

    # freezing cost
    def run(names, pos0):
        for n, ln in model.lns():
            if n in names:
                s_rest, s0, s_all = consts[n]
                ln.frozen = (s_rest, s0) if pos0 else (s_all, s_all)
        out = loss_on(model, Xv, Yv)
        for _, ln in model.lns():
            ln.frozen = None
        return out

    frows = []
    all_names = [n for n, _ in model.lns()]
    for n in all_names + ["ALL"]:
        names = all_names if n == "ALL" else [n]
        for pos0 in (False, True):
            l = run(names, pos0)
            frows.append(dict(frozen=n, separate_pos0=pos0, loss=l, dloss=l - base, rel=(l - base) / base))
    fr = pd.DataFrame(frows)
    fr["base_loss"] = base
    fr.to_csv(os.path.join(OUT, "ln_freeze.csv"), index=False)
    print(fr.pivot(index="frozen", columns="separate_pos0", values="dloss").round(4).to_string(), flush=True)
    print("LN DONE", flush=True)


if __name__ == "__main__":
    main()
