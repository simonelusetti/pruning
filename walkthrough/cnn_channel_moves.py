"""
Channel moves for a small CNN, worked on real numbers.

A tiny CNN is trained on sklearn's 8x8 digits (no download needed):
    conv1 1->16 (3x3, zero padding 1) -> ReLU -> conv2 16->32 (3x3, padding 1) -> ReLU
    -> global average pool -> fc 32->10
Then the channels of conv1 (which feed conv2) are pruned by hand with the
CNN versions of Veronica's unit moves, and every fold is checked against the
substitution it is supposed to reproduce.

Sections
  1. what a channel is, and what removing one frees
  2. dead channels
  3. replacing a channel by a constant: three ways to fold it (the padding problem)
  4. merging two channels
  5. summary table of validation-loss changes

Run:  python cnn_channel_moves.py      (CPU, ~20 s)
Output: cnn_channel_moves_output.txt next to this file
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


# ----------------------------------------------------------------------------
# data and model
# ----------------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)
T = lambda a: torch.tensor(a)
Xtr, Xva, Xte = T(Xtr), T(Xva), T(Xte)
ytr, yva, yte = T(ytr).long(), T(yva).long(), T(yte).long()

C1, C2 = 16, 32
P = {
    "W1": torch.randn(C1, 1, 3, 3) * 0.5, "b1": torch.zeros(C1),
    "W2": torch.randn(C2, C1, 3, 3) * (2 / (C1 * 9)) ** 0.5, "b2": torch.zeros(C2),
    "Wf": torch.randn(10, C2) * 0.2, "bf": torch.zeros(10),
}


def forward(P, x, a1_override=None, bias2_map=None, return_a1=False):
    """bias2_map: optional extra bias for conv2, shape (C2, 8, 8) (per position)."""
    a1 = F.relu(F.conv2d(x, P["W1"], P["b1"], padding=1))
    if a1_override is not None:
        a1 = a1_override(a1)
    z2 = F.conv2d(a1, P["W2"], P["b2"], padding=1)
    if bias2_map is not None:
        z2 = z2 + bias2_map
    a2 = F.relu(z2)
    out = a2.mean((2, 3)) @ P["Wf"].T + P["bf"]
    return (out, a1) if return_a1 else out


params = [v.requires_grad_(True) for v in P.values()]
opt = torch.optim.Adam(params, lr=3e-3)
for ep in range(1500):
    opt.zero_grad()
    loss = F.cross_entropy(forward(P, Xtr), ytr)
    loss.backward()
    opt.step()
P = {k: v.detach() for k, v in P.items()}


def per_sample_loss(P, Xs, ys, **kw):
    with torch.no_grad():
        return F.cross_entropy(forward(P, Xs, **kw), ys, reduction="none")


def paired(P2, kw2=None, P1=None, kw1=None, split=(Xva, yva)):
    a = per_sample_loss(P2, *split, **(kw2 or {}))
    b = per_sample_loss(P1 or P, *split, **(kw1 or {}))
    d = (a - b).double()
    return float(d.mean()), float(d.std() / d.numel() ** 0.5)


def acc(P_, Xs, ys, **kw):
    with torch.no_grad():
        return float((forward(P_, Xs, **kw).argmax(1) == ys).float().mean())


n_params = sum(v.numel() for v in P.values())
say("=" * 78)
say("THE NETWORK")
say("=" * 78)
say(f"conv1 1->{C1}, conv2 {C1}->{C2} (3x3, zero padding), GAP, fc {C2}->10;  {n_params} parameters")
say(f"train/val/test = {len(ytr)}/{len(yva)}/{len(yte)} digits (8x8 images)")
say(f"val loss {per_sample_loss(P, Xva, yva).mean():.4f}, val acc {acc(P, Xva, yva):.3f}, "
    f"test acc {acc(P, Xte, yte):.3f}")

# ----------------------------------------------------------------------------
# 1. channels
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("1. WHAT A conv1 CHANNEL IS")
say("=" * 78)
with torch.no_grad():
    _, A1 = forward(P, Xtr, return_a1=True)            # (n, 16, 8, 8)
fan_in = 1 * 3 * 3 + 1
fan_out = C2 * 3 * 3
say(f"channel i of conv1 = one 3x3 filter W1[i] (+ bias b1[i]) slid over the image -> one 8x8 map a_i(x)")
say(f"conv2 reads channel i through the slices W2[:, i] (a 3x3 kernel for each of its {C2} filters)")
say(f"removing channel i frees fan-in {fan_in} + fan-out {fan_out} = {fan_in + fan_out} parameters")

# ----------------------------------------------------------------------------
# 2. dead channels
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("2. DEAD CHANNELS (ReLU output 0 at every pixel of every training image)")
say("=" * 78)
frac_zero = (A1 == 0).float().mean((0, 2, 3))
for i in range(C1):
    say(f"  channel {i:2d}: fraction of zero pixels {float(frac_zero[i]):.3f}   "
        f"mean {float(A1[:, i].mean()):.3f}   std {float(A1[:, i].std()):.3f}")
dead = [i for i in range(C1) if float(A1[:, i].abs().max()) == 0.0]
say(f"dead channels: {dead if dead else 'none in this network'}")


def remove_channel(P_, i):
    """Delete conv1 channel i: zero its filter and bias and its slices in conv2."""
    Q = {k: v.clone() for k, v in P_.items()}
    Q["W1"][i] = 0
    Q["b1"][i] = 0
    Q["W2"][:, i] = 0
    return Q


# ----------------------------------------------------------------------------
# 3. constant replacement and the padding problem
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("3. REPLACING A CHANNEL BY A CONSTANT")
say("=" * 78)
std = A1.std((0, 2, 3)) / (A1.mean((0, 2, 3)).abs() + 1e-6)
j = int(torch.argmin(A1.std((0, 2, 3)) + 1e3 * (A1.amax((0, 2, 3)) == 0).float()))
c = float(A1[:, j].mean())
say(f"least-varying live channel: {j}, mean {c:.4f}, std {float(A1[:, j].std()):.4f}")


def substitute_const(a1, i=j, val=c):
    a1 = a1.clone()
    a1[:, i] = val
    return a1


ones = torch.zeros(1, C1, 8, 8)
ones[0, j] = 1.0
# what conv2 does with a map that is 1 everywhere in channel j (and 0 elsewhere), zero padding included
resp = F.conv2d(ones, P["W2"], None, padding=1)[0]      # (C2, 8, 8)
say("\nconv2 applied to a constant map of 1s in channel j, for output filter 0 (zero padding):")
for r in resp[0]:
    say("   " + " ".join(f"{float(v):+.3f}" for v in r))
say("interior pixels all get sum(W2[0, j]) =", f"{float(P['W2'][0, j].sum()):+.3f}",
    "; border pixels miss the taps that fall on the padding")

# (a) scalar fold: b2 += c * sum of kernel  -> exact only in the interior
Qa = remove_channel(P, j)
Qa["b2"] = Qa["b2"] + c * P["W2"][:, j].sum((1, 2))
# (b) border-aware fold: the exact response c * resp, but it only takes 9 distinct
#     values per output filter (4 corners, 4 edges, interior) for a 3x3 kernel
Qb = remove_channel(P, j)
B_b = c * resp
distinct = len(torch.unique(B_b[0].round(decimals=6)))
# (c) per-position MEAN map instead of a scalar: closer projection, folded as a full bias map
m_map = A1[:, j].mean(0)                                  # (8, 8): E over images, per pixel
mm = torch.zeros(1, C1, 8, 8)
mm[0, j] = m_map
B_c = F.conv2d(mm, P["W2"], None, padding=1)[0]
Qc = remove_channel(P, j)


def substitute_map(a1, i=j, M=m_map):
    a1 = a1.clone()
    a1[:, i] = M
    return a1


with torch.no_grad():
    ref_sub = forward(P, Xva, a1_override=substitute_const)          # what the move should give
    ref_map = forward(P, Xva, a1_override=substitute_map)
    err_a = float((forward(Qa, Xva) - ref_sub).abs().max())
    err_b = float((forward(Qb, Xva, bias2_map=B_b) - ref_sub).abs().max())
    err_c = float((forward(Qc, Xva, bias2_map=B_c) - ref_map).abs().max())
say(f"\nfold check: max |logit difference| between the folded network and the network with")
say(f"channel {j} literally replaced by the constant / mean map")
say(f"  (a) scalar bias   b2 += c * sum(W2[:, j])         error {err_a:.2e}   extra numbers stored: 0")
say(f"  (b) border-aware  9 values per conv2 filter       error {err_b:.2e}   extra numbers stored: "
    f"{C2 * 9} once per layer ({distinct} distinct values found per filter)")
say(f"  (c) per-pixel mean map, full bias map             error {err_c:.2e}   extra numbers stored: "
    f"{C2 * 64} once per layer")

# ----------------------------------------------------------------------------
# 4. merging two channels
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("4. MERGING TWO CHANNELS")
say("=" * 78)
flat = A1.permute(1, 0, 2, 3).reshape(C1, -1)            # each channel = all pixels of all images
live = [i for i in range(C1) if float(flat[i].abs().max()) > 0 and i != j]
Cm = torch.corrcoef(flat[live])
Cm.fill_diagonal_(0)
k = int(Cm.abs().argmax())
p, q = live[k // len(live)], live[k % len(live)]
say(f"most correlated pair among live channels: {p} and {q}, corr over pixels and images = "
    f"{float(Cm.abs().max()):.4f}")
xp, xq = flat[p] - flat[p].mean(), flat[q] - flat[q].mean()
alpha = float((xp * xq).mean() / xq.pow(2).mean())
beta = float(flat[p].mean() - alpha * flat[q].mean())
say(f"least squares over all pixels:  a_{p} ~ {alpha:.3f} * a_{q} + {beta:+.4f}")
Qm = remove_channel(P, p)
Qm["W2"][:, q] = P["W2"][:, q] + alpha * P["W2"][:, p]     # the alpha part: exact
onesp = torch.zeros(1, C1, 8, 8)
onesp[0, p] = 1.0
B_m = beta * F.conv2d(onesp, P["W2"], None, padding=1)[0]  # the beta part: border-aware constant
Qm_scalar = {k_: v.clone() for k_, v in Qm.items()}
Qm_scalar["b2"] = Qm["b2"] + beta * P["W2"][:, p].sum((1, 2))


def substitute_merge(a1):
    a1 = a1.clone()
    a1[:, p] = alpha * a1[:, q] + beta
    return a1


with torch.no_grad():
    ref = forward(P, Xva, a1_override=substitute_merge)
    e1 = float((forward(Qm, Xva, bias2_map=B_m) - ref).abs().max())
    e2 = float((forward(Qm_scalar, Xva) - ref).abs().max())
say(f"fold check vs the literal substitution a_{p} := alpha*a_{q}+beta:")
say(f"  alpha into W2[:, {q}], beta border-aware   error {e1:.2e}")
say(f"  alpha into W2[:, {q}], beta as scalar bias  error {e2:.2e}")

# ----------------------------------------------------------------------------
# 5. summary
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say(f"5. VALIDATION-LOSS CHANGE OF EACH MOVE (paired against the original, n = {len(yva)})")
say("=" * 78)
rows = [
    (f"zero channel {j} (classical)", remove_channel(P, j), {}),
    (f"constant {j}, scalar fold", Qa, {}),
    (f"constant {j}, border-aware fold", Qb, {"bias2_map": B_b}),
    (f"mean map {j}, per-pixel fold", Qc, {"bias2_map": B_c}),
    (f"zero channel {p} (classical)", remove_channel(P, p), {}),
    (f"merge {p}->{q}, scalar beta", Qm_scalar, {}),
    (f"merge {p}->{q}, border-aware beta", Qm, {"bias2_map": B_m}),
]
say(f"{'move':36s} {'freed':>5s} {'dLoss':>10s} {'SE':>8s} {'val acc':>8s}")
for name, Q, kw in rows:
    m_, se_ = paired(Q, kw)
    say(f"{name:36s} {fan_in + fan_out:5d} {m_:+10.5f} {se_:8.5f} {acc(Q, Xva, yva, **kw):8.3f}")

with open(os.path.join(HERE, "cnn_channel_moves_output.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
