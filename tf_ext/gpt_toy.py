"""
A toy GPT trained as a real one: character-level language model on Tiny Shakespeare.

Architecture (GPT-2 / nanoGPT layout, toy sizes: d = 64, 4 heads of 16, MLP 256, L = 2,
context T = 32):
    x = W_E[tok] + W_pos[t]
    for each block l:
        x = x + Attn_l(LN1_l(x))     q, k, v = h W_Q + b_Q, h W_K + b_K, h W_V + b_V   (causal)
                                     out = sum_heads A v W_O + b_O
        x = x + MLP_l(LN2_l(x))      W_2 GELU(W_1 h + b_1) + b_2      (exact, erf GELU)
    logits = LN_f(x) W_U + b_U       (untied: W_U is its own d x V matrix)
    LN(x) = gamma * (x - mean x) / sqrt(var x + 1e-5) + beta

Nothing is removed or folded here: gamma, beta and every bias are trained as in GPT.
The embedding is untied (as in LLaMA-style models; GPT-2 ties W_U = W_E^T): with a tie
the exact LayerNorm folds and the stream move would change the two uses of W_E
differently.  The tied models are kept in results_gpt_tied/.
Folds and moves happen only inside the pruning runs, on a copy.

Data: Karpathy's tinyshakespeare (data/shakespeare/input.txt, 1 115 394 characters,
vocabulary 65), split 80 / 10 / 10 into train / val / test text, so that the pruning
runs have held-out text (val -> valA, valB) and a test split never used for anything.
Loss: next-character cross-entropy at every position.

Training: nanoGPT's shakespeare_char recipe, scaled to the toy: AdamW, lr 1e-3 with
100 warm-up steps and cosine decay to 1e-4, betas (0.9, 0.99), weight decay 0.1 on
tensors with >= 2 dimensions (matrices, W_E, W_pos, W_U) and none on biases, gamma, beta,
gradient clipping at 1.0, batch 64 random windows, dropout 0.  Init as nanoGPT:
N(0, 0.02), W_O and W_2 at 0.02 / sqrt(2 L), biases 0, gamma 1, beta 0.

Usage:  SEED=0 STEPS=10000 L=2 python3 gpt_toy.py
        writes results_gpt/L<L>_seed<k>/{base_state.pt, step<n>.pt, hist.csv}
"""
from __future__ import annotations

import math
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(os.path.dirname(HERE), "data", "shakespeare", "input.txt")
EPS = 1e-5


class GPTConfig:
    def __init__(self, V=65, T=32, d=64, H=4, d_mlp=256, L=2):
        self.V, self.T, self.d, self.H, self.d_mlp, self.L = V, T, d, H, d_mlp, L
        self.dh = d // H

    def as_dict(self):
        return dict(V=self.V, T=self.T, d=self.d, H=self.H, d_mlp=self.d_mlp, L=self.L)


# ---- data ---------------------------------------------------------------------------

def load_text(path=TEXT):
    """-> dict split -> 1-D LongTensor of character ids, and the vocabulary."""
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    table = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([table[c] for c in text], dtype=torch.long)
    n = len(ids)
    a, b = int(0.8 * n), int(0.9 * n)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}, chars


def windows(ids, T, n=None, seed=0):
    """Non-overlapping windows (X, Y) of length T (Y = X shifted by one), optionally a
    fixed random subset of n of them."""
    m = (len(ids) - 1) // T
    X = ids[: m * T].view(m, T)
    Y = ids[1: m * T + 1].view(m, T)
    if n is not None and n < m:
        sel = torch.randperm(m, generator=torch.Generator().manual_seed(seed))[:n]
        X, Y = X[sel], Y[sel]
    return X, Y


def random_batch(ids, T, B, g):
    i = torch.randint(0, len(ids) - T - 1, (B,), generator=g)
    return torch.stack([ids[j:j + T] for j in i]), torch.stack([ids[j + 1:j + T + 1] for j in i])


# ---- model -------------------------------------------------------------------------------

def init_params(cfg: GPTConfig, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = lambda *s, std=0.02: torch.randn(*s, generator=g) * std
    L, H, d, dh, dm = cfg.L, cfg.H, cfg.d, cfg.dh, cfg.d_mlp
    res = 0.02 / math.sqrt(2 * L)
    return {
        "W_E": n(cfg.V, d), "W_pos": n(cfg.T, d),
        "ln1_g": torch.ones(L, d), "ln1_b": torch.zeros(L, d),
        "W_Q": n(L, H, d, dh), "W_K": n(L, H, d, dh), "W_V": n(L, H, d, dh),
        "b_Q": torch.zeros(L, H, dh), "b_K": torch.zeros(L, H, dh), "b_V": torch.zeros(L, H, dh),
        "W_O": n(L, H, dh, d, std=res), "b_O": torch.zeros(L, d),
        "ln2_g": torch.ones(L, d), "ln2_b": torch.zeros(L, d),
        "W_1": n(L, dm, d), "b_1": torch.zeros(L, dm),
        "W_2": n(L, d, dm, std=res), "b_2": torch.zeros(L, d),
        "lnf_g": torch.ones(d), "lnf_b": torch.zeros(d),
        "W_U": n(d, cfg.V), "b_U": torch.zeros(cfg.V),
    }


def layer_norm(x, g, b):
    c = x - x.mean(-1, keepdim=True)
    return g * c / torch.sqrt(c.pow(2).mean(-1, keepdim=True) + EPS) + b


def forward(P: dict, cfg: GPTConfig, X: torch.Tensor, cache: bool = False):
    """X: (n, T) token ids -> logits (n, T, V) [, cache].  cache['x_in'] lists the stream
    at the input of every LayerNorm (ln1, ln2 of each block, then ln_f)."""
    T = X.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    x = P["W_E"][X] + P["W_pos"][:T]
    c = {"x_in": [], "attn": [], "ho": [], "act": []} if cache else None
    for l in range(cfg.L):
        h = layer_norm(x, P["ln1_g"][l], P["ln1_b"][l])
        q = torch.einsum("btd,hde->bhte", h, P["W_Q"][l]) + P["b_Q"][l][:, None]
        k = torch.einsum("btd,hde->bhte", h, P["W_K"][l]) + P["b_K"][l][:, None]
        v = torch.einsum("btd,hde->bhte", h, P["W_V"][l]) + P["b_V"][l][:, None]
        sc = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(cfg.dh)
        A = torch.softmax(sc.masked_fill(~causal, float("-inf")), -1)
        ho = torch.einsum("bhts,bhse,hed->bhtd", A, v, P["W_O"][l])
        if cache:
            c["x_in"].append(x); c["attn"].append(A); c["ho"].append(ho)
        x = x + ho.sum(1) + P["b_O"][l]
        h = layer_norm(x, P["ln2_g"][l], P["ln2_b"][l])
        a = F.gelu(h @ P["W_1"][l].T + P["b_1"][l])
        if cache:
            c["x_in"].append(x); c["act"].append(a)
        x = x + a @ P["W_2"][l].T + P["b_2"][l]
    if cache:
        c["x_in"].append(x)
    z = layer_norm(x, P["lnf_g"], P["lnf_b"]) @ P["W_U"] + P["b_U"]
    return (z, c) if cache else z


def n_params(P: dict) -> int:
    return sum(v.numel() for v in P.values())


@torch.no_grad()
def evaluate(P, cfg, X, Y, bs=512):
    tot, correct, n = 0.0, 0, 0
    for i in range(0, len(X), bs):
        z = forward(P, cfg, X[i:i + bs])
        tot += float(F.cross_entropy(z.flatten(0, 1), Y[i:i + bs].flatten(), reduction="sum"))
        correct += int((z.argmax(-1) == Y[i:i + bs]).sum())
        n += Y[i:i + bs].numel()
    return tot / n, correct / n


# ---- training ------------------------------------------------------------------------------

def lr_at(it, steps, lr, warmup=100, min_lr=1e-4):
    if it < warmup:
        return lr * (it + 1) / warmup
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * (it - warmup) / max(1, steps - warmup)))


def train(seed=0, L=2, steps=10000, lr=1e-3, wd=0.1, batch=64, log_every=250, save_every=2500, out=None):
    cfg = GPTConfig(L=L)
    data, chars = load_text()
    assert len(chars) == cfg.V
    Xva, Yva = windows(data["val"], cfg.T, n=1024, seed=0)       # fixed monitoring set
    torch.manual_seed(seed)
    P = {k: v.requires_grad_(True) for k, v in init_params(cfg, seed).items()}
    decay = [v for v in P.values() if v.dim() >= 2]
    no_decay = [v for v in P.values() if v.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.99))
    g = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    print(f"L={L} seed {seed}: {n_params(P)} parameters; {len(data['train'])} train characters", flush=True)
    hist, t0, run = [], time.time(), 0.0
    for it in range(steps + 1):
        if it % log_every == 0 or it == steps:
            va = evaluate(P, cfg, Xva, Yva)
            hist.append(dict(step=it, train_loss=run, val_loss=va[0], val_acc=va[1]))
            print(f"L={L} seed {seed} step {it:6d}  train(avg) {run:.4f}  val {va[0]:.4f} acc {va[1]:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if it % save_every == 0 and 0 < it < steps:
            torch.save({k: v.detach().clone() for k, v in P.items()}, os.path.join(out, f"step{it}.pt"))
        if it == steps:
            break
        for gr in opt.param_groups:
            gr["lr"] = lr_at(it, steps, lr)
        X, Y = random_batch(data["train"], cfg.T, batch, g)
        loss = F.cross_entropy(forward(P, cfg, X).flatten(0, 1), Y.flatten())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
    base = {"params": {k: v.detach().clone() for k, v in P.items()}, "cfg": cfg.as_dict(), "chars": chars,
            "seed": seed, "steps": steps, "lr": lr, "wd": wd, "batch": batch}
    torch.save(base, os.path.join(out, "base_state.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out, "hist.csv"), index=False)
    res = {sp: evaluate(base["params"], cfg, *windows(data[sp], cfg.T)) for sp in ("train", "val", "test")}
    print(f"DONE L={L} seed {seed}: " + "  ".join(f"{sp} loss {v[0]:.4f} acc {v[1]:.3f}" for sp, v in res.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    seed, L = int(os.environ.get("SEED", "0")), int(os.environ.get("L", "2"))
    train(seed=seed, L=L, steps=int(os.environ.get("STEPS", "10000")),
          out=os.path.join(os.environ.get("GPT_RESULTS", os.path.join(HERE, "results_gpt")), f"L{L}_seed{seed}"))
