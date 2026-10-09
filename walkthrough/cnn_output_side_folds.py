"""
Output-side folds for CNN channel moves.

Idea: instead of approximating the CHANNEL (input side of conv2) and then pushing
the approximation through the padded convolution, approximate the channel's
CONTRIBUTION to conv2's output directly, by least squares, within a family that
folds exactly into conv2's existing parameters (bias, or the kernel of a kept
channel).  The padding is then handled automatically, because the contribution
already contains it.

Same tiny CNN and data as cnn_channel_moves.py (8x8 digits, conv1 1->16,
conv2 16->32, 3x3 zero padding, GAP, fc).

Folds compared for removing conv1 channel j:
  constant family (channel replaced by something input-independent)
    in-scalar     channel -> its mean c, bias += c * sum(kernel)          (naive MLP fold)
    in-border     channel -> c, exact border-aware bias (9 values/filter)  (previous demo)
    out-scalar    contribution -> its mean per conv2 filter                 0 extra numbers
    out-region    contribution -> its mean per filter and border region     9 per filter
    out-pixel     contribution -> its mean per filter and pixel (= mean map) 64 per filter
  merge family (channel i replaced using a kept channel i')
    pointwise     a_i ~ alpha a_i' + beta, folded exactly (previous demo)
    spatial       contribution of i ~ V_o * a_i' + b_o, a 3x3 kernel V_o per conv2 filter,
                  fitted by least squares, folded as W2[o, i'] += V_o, b2[o] += b_o

Run:  python cnn_output_side_folds.py     (CPU, ~1 min)
Output: cnn_output_side_folds_output.txt next to this file
"""
import os

import numpy as np
import torch
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


# ---------------------------------------------------------------- data, model (as before)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)
T = lambda a: torch.tensor(a)
Xtr, Xva, Xte = T(Xtr), T(Xva), T(Xte)
ytr, yva, yte = T(ytr).long(), T(yva).long(), T(yte).long()
C1, C2 = 16, 32
P = {"W1": torch.randn(C1, 1, 3, 3) * 0.5, "b1": torch.zeros(C1),
     "W2": torch.randn(C2, C1, 3, 3) * (2 / (C1 * 9)) ** 0.5, "b2": torch.zeros(C2),
     "Wf": torch.randn(10, C2) * 0.2, "bf": torch.zeros(10)}


def forward(P, x, bias2_map=None):
    a1 = F.relu(F.conv2d(x, P["W1"], P["b1"], padding=1))
    z2 = F.conv2d(a1, P["W2"], P["b2"], padding=1)
    if bias2_map is not None:
        z2 = z2 + bias2_map
    return F.relu(z2).mean((2, 3)) @ P["Wf"].T + P["bf"]


def conv1_out(P, x):
    return F.relu(F.conv2d(x, P["W1"], P["b1"], padding=1))


params = [v.requires_grad_(True) for v in P.values()]
opt = torch.optim.Adam(params, lr=3e-3)
for _ in range(1500):
    opt.zero_grad()
    F.cross_entropy(forward(P, Xtr), ytr).backward()
    opt.step()
P = {k: v.detach() for k, v in P.items()}


def dloss(Q, bias2_map=None):
    with torch.no_grad():
        a = F.cross_entropy(forward(Q, Xva, bias2_map), yva, reduction="none")
        b = F.cross_entropy(forward(P, Xva), yva, reduction="none")
        d = (a - b).double()
        acc = float((forward(Q, Xva, bias2_map).argmax(1) == yva).float().mean())
    return float(d.mean()), float(d.std() / d.numel() ** 0.5), acc


def removed(i):
    Q = {k: v.clone() for k, v in P.items()}
    Q["W1"][i] = 0
    Q["b1"][i] = 0
    Q["W2"][:, i] = 0
    return Q


def contribution(A, i, K=None):
    """Contribution of conv1 channel i to every conv2 filter: (n, C2, 8, 8)."""
    K = P["W2"][:, i:i + 1] if K is None else K
    return F.conv2d(A[:, i:i + 1], K, None, padding=1)


# border region of each pixel for a 3x3 kernel with padding 1: 0 = top/left edge, 1 = interior, 2 = bottom/right
reg = torch.tensor([0] + [1] * 6 + [2])
region_id = (reg[:, None] * 3 + reg[None, :])            # (8, 8), values 0..8

with torch.no_grad():
    A_tr = conv1_out(P, Xtr)

say(f"val loss {F.cross_entropy(forward(P, Xva), yva):.4f}, val acc "
    f"{float((forward(P, Xva).argmax(1) == yva).float().mean()):.3f}")

# ---------------------------------------------------------------- constant family
say("")
say("=" * 78)
say("CONSTANT FOLDS (channel j removed, its effect replaced by something input-independent)")
say("=" * 78)
say(f"{'channel':>7s} {'fold':12s} {'extra':>6s} {'dLoss':>9s} {'SE':>7s} {'val acc':>8s}")
results = {}
for j in range(C1):
    c = float(A_tr[:, j].mean())
    Cj = contribution(A_tr, j)                              # (n, C2, 8, 8)
    ones = torch.zeros(1, C1, 8, 8)
    ones[0, j] = 1.0
    S = F.conv2d(ones, P["W2"], None, padding=1)[0]          # (C2, 8, 8)
    folds = {}
    Q = removed(j); Q["b2"] = Q["b2"] + c * P["W2"][:, j].sum((1, 2)); folds["in-scalar"] = (Q, None, 0)
    folds["in-border"] = (removed(j), c * S, 9 * C2)
    Q = removed(j); Q["b2"] = Q["b2"] + Cj.mean((0, 2, 3)); folds["out-scalar"] = (Q, None, 0)
    Bm = torch.zeros(C2, 8, 8)
    mean_o_p = Cj.mean(0)                                    # (C2, 8, 8)
    for r in range(9):
        msk = region_id == r
        Bm[:, msk] = mean_o_p[:, msk].mean(1, keepdim=True)
    folds["out-region"] = (removed(j), Bm, 9 * C2)
    folds["out-pixel"] = (removed(j), mean_o_p, 64 * C2)
    for name, (Q, B, extra) in folds.items():
        results[(j, name)] = dloss(Q, B) + (extra,)

names = ["in-scalar", "in-border", "out-scalar", "out-region", "out-pixel"]
for j in range(C1):
    for name in names:
        m, se, acc, extra = results[(j, name)]
        say(f"{j:7d} {name:12s} {extra:6d} {m:+9.4f} {se:7.4f} {acc:8.3f}")
say("\nmedian dLoss over the 16 channels:")
for name in names:
    say(f"  {name:12s} {np.median([results[(j, name)][0] for j in range(C1)]):+.4f}   "
        f"(best of the five on {sum(min(names, key=lambda n: results[(j, n)][0]) == name for j in range(C1))} channels)")

# ---------------------------------------------------------------- merge family
say("")
say("=" * 78)
say("MERGES: pointwise (alpha, beta) vs spatial (3x3 kernel per conv2 filter), fitted output-side")
say("=" * 78)
flat = A_tr.permute(1, 0, 2, 3).reshape(C1, -1)
# patches of every kept channel: (n*64, 9) per channel, zero padding included
patches = F.unfold(A_tr, 3, padding=1)                       # (n, C1*9, 64)
patches = patches.view(A_tr.shape[0], C1, 9, 64).permute(1, 0, 3, 2).reshape(C1, -1, 9)


def pointwise_merge(i, i2):
    xi, xj = flat[i] - flat[i].mean(), flat[i2] - flat[i2].mean()
    alpha = float((xi * xj).mean() / xj.pow(2).mean())
    beta = float(flat[i].mean() - alpha * flat[i2].mean())
    Q = removed(i)
    Q["W2"][:, i2] = P["W2"][:, i2] + alpha * P["W2"][:, i]
    one = torch.zeros(1, C1, 8, 8)
    one[0, i] = 1.0
    B = beta * F.conv2d(one, P["W2"], None, padding=1)[0]
    return Q, B


def spatial_merge(i, i2):
    """Least squares, per conv2 filter o: contribution_o of channel i ~ V_o (*) a_i2 + b_o."""
    target = contribution(A_tr, i).permute(1, 0, 2, 3).reshape(C2, -1)      # (C2, n*64)
    Z = torch.cat([patches[i2], torch.ones(patches.shape[1], 1)], 1).double()  # (n*64, 10)
    sol = torch.linalg.lstsq(Z, target.T.double()).solution                  # (10, C2)
    V = sol[:9].T.float().reshape(C2, 3, 3)
    b = sol[9].float()
    Q = removed(i)
    Q["W2"][:, i2] = P["W2"][:, i2] + V
    Q["b2"] = Q["b2"] + b
    resid = target.T.double() - Z @ sol
    r2 = 1 - float(resid.pow(2).sum() / (target - target.mean(1, keepdim=True)).pow(2).sum())
    return Q, r2


Cm = torch.corrcoef(flat)
Cm.fill_diagonal_(0)
say(f"{'remove':>6s} {'partner':>8s} {'pixel corr':>10s} {'method':10s} {'dLoss':>9s} {'SE':>7s} {'val acc':>8s} {'R2 (spatial)':>12s}")
summary = []
for i in range(C1):
    pw_partner = int(Cm[i].abs().argmax())
    # spatial: best partner by R2 of the output-side regression
    best = None
    for i2 in range(C1):
        if i2 == i:
            continue
        _, r2 = spatial_merge(i, i2)
        if best is None or r2 > best[1]:
            best = (i2, r2)
    Qp, Bp = pointwise_merge(i, pw_partner)
    mp = dloss(Qp, Bp)
    Qs, r2 = spatial_merge(i, best[0])
    ms = dloss(Qs)
    summary.append((mp[0], ms[0]))
    say(f"{i:6d} {pw_partner:8d} {float(Cm[i, pw_partner]):10.3f} {'pointwise':10s} {mp[0]:+9.4f} {mp[1]:7.4f} {mp[2]:8.3f}")
    say(f"{'':6s} {best[0]:8d} {float(Cm[i, best[0]]):10.3f} {'spatial':10s} {ms[0]:+9.4f} {ms[1]:7.4f} {ms[2]:8.3f} {r2:12.3f}")
pm = np.array(summary)
say(f"\nmedian dLoss: pointwise {np.median(pm[:, 0]):+.4f}, spatial {np.median(pm[:, 1]):+.4f}; "
    f"spatial better on {(pm[:, 1] < pm[:, 0]).sum()} of {C1} channels")

with open(os.path.join(HERE, "cnn_output_side_folds_output.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
