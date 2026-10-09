"""
Weight-space search for shifted channels in conv1 of the digits CNN.

Convolution is translation-equivariant: if filter i is filter i' moved by s
pixels (times alpha > 0, with bias b_i = alpha * b_i'), then map a_i is map a_i'
moved by s pixels, up to the image border.  So shifted channels can be looked for
in the WEIGHTS, without data: for each pair (i, i') and each shift s in the 3x3
window, fit alpha by least squares and measure how much of filter i is left
unexplained.  The data-driven spatial-merge partners (cnn_output_side_folds.py)
are printed next to it for comparison.

Run: python cnn_shift_search.py   (retrains the same network, ~30 s)
"""
import numpy as np, torch, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr).long()
C1, C2 = 16, 32
P = {"W1": torch.randn(C1, 1, 3, 3) * 0.5, "b1": torch.zeros(C1),
     "W2": torch.randn(C2, C1, 3, 3) * (2 / (C1 * 9)) ** 0.5, "b2": torch.zeros(C2),
     "Wf": torch.randn(10, C2) * 0.2, "bf": torch.zeros(10)}
def forward(P, x):
    a1 = F.relu(F.conv2d(x, P["W1"], P["b1"], padding=1))
    return F.relu(F.conv2d(a1, P["W2"], P["b2"], padding=1)).mean((2, 3)) @ P["Wf"].T + P["bf"]
ps = [v.requires_grad_(True) for v in P.values()]
opt = torch.optim.Adam(ps, lr=3e-3)
for _ in range(1500):
    opt.zero_grad(); F.cross_entropy(forward(P, Xtr), ytr).backward(); opt.step()
W = P["W1"].detach()[:, 0]          # (16, 3, 3)
b = P["b1"].detach()

def shifted(K, dy, dx):
    """K moved by (dy, dx) inside its 3x3 window; taps pushed outside are lost."""
    out = torch.zeros_like(K)
    for r in range(3):
        for c in range(3):
            rr, cc = r + dy, c + dx
            if 0 <= rr < 3 and 0 <= cc < 3:
                out[rr, cc] = K[r, c]
    return out

data_partner = {0: 6, 1: 2, 2: 6, 3: 4, 4: 6, 5: 7, 6: 4, 7: 5, 8: 10, 9: 14, 10: 8, 11: 10, 12: 4, 13: 6, 14: 4, 15: 13}
print("conv1 filters (3x3) and biases")
for i in range(C1):
    print(f"  {i:2d}: " + " | ".join(" ".join(f"{float(v):+.2f}" for v in row) for row in W[i]) + f"   b={float(b[i]):+.2f}")
print(f"\n{'i':>2s}  best weight-space match: partner, shift (dy,dx), alpha, unexplained %, bias check b_i vs alpha*b_i'   | data partner")
for i in range(C1):
    best = None
    for j in range(C1):
        if j == i: continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                S = shifted(W[j], dy, dx)
                den = float((S * S).sum())
                if den == 0: continue
                alpha = float((W[i] * S).sum()) / den
                if alpha <= 0: continue
                res = float(((W[i] - alpha * S) ** 2).sum() / (W[i] ** 2).sum())
                if best is None or res < best[0]:
                    best = (res, j, dy, dx, alpha)
    res, j, dy, dx, alpha = best
    print(f"{i:2d}  {j:2d}  ({dy:+d},{dx:+d})  alpha {alpha:5.2f}  {100*res:5.1f}%   b {float(b[i]):+.2f} vs {alpha*float(b[j]):+.2f}"
          f"   | {data_partner[i]:2d}{'  same' if data_partner[i]==j else ''}")
