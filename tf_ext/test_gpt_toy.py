"""Checks for the toy GPT.  Run: python3 test_gpt_toy.py  (or pytest)"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpt_toy as G  # noqa: E402

CFG = G.GPTConfig()
P = G.init_params(CFG, seed=0)
for k, v in P.items():                      # non-trivial gamma, beta, biases
    if k.endswith(("_g", "_b")) or k.startswith("b_"):
        P[k] = v + 0.1 * torch.randn_like(v)
X = torch.randint(0, 65, (16, 32))


def test_parameter_count():
    # per block 49 984, W_E 65 x 64 = 4 160, W_U 64 x 65 = 4 160, W_pos 32 x 64 = 2 048, LN_f 128, b_U 65
    assert G.n_params(P) == 2 * 49984 + 2 * 4160 + 2048 + 128 + 65 == 110529


def test_layer_norm_matches_torch():
    x = torch.randn(5, 3, 64)
    g, b = torch.randn(64), torch.randn(64)
    assert torch.allclose(G.layer_norm(x, g, b), F.layer_norm(x, (64,), g, b, eps=G.EPS), atol=1e-5)


def test_attention_matches_torch():
    """Block-0 attention output against torch's scaled_dot_product_attention."""
    _, c = G.forward(P, CFG, X, cache=True)
    h = G.layer_norm(c["x_in"][0], P["ln1_g"][0], P["ln1_b"][0])
    proj = lambda W, b: (torch.einsum("btd,hde->bhte", h, W[0]) + b[0][:, None])
    a = F.scaled_dot_product_attention(proj(P["W_Q"], P["b_Q"]), proj(P["W_K"], P["b_K"]),
                                       proj(P["W_V"], P["b_V"]), is_causal=True)
    ref = torch.einsum("bhte,hed->btd", a, P["W_O"][0]) + P["b_O"][0]
    assert torch.allclose(c["x_in"][1] - c["x_in"][0], ref, atol=1e-5)


def test_causal():
    z1 = G.forward(P, CFG, X)
    X2 = X.clone(); X2[:, 20] = (X2[:, 20] + 1) % 65
    z2 = G.forward(P, CFG, X2)
    assert torch.allclose(z1[:, :20], z2[:, :20], atol=1e-6) and not torch.allclose(z1[:, 20:], z2[:, 20:])


def test_untied_unembedding():
    _, c = G.forward(P, CFG, X, cache=True)
    z = G.forward(P, CFG, X)
    ref = G.layer_norm(c["x_in"][-1], P["lnf_g"], P["lnf_b"]) @ P["W_U"] + P["b_U"]
    assert torch.allclose(z, ref)
    assert P["W_U"].data_ptr() != P["W_E"].data_ptr() and not torch.allclose(P["W_U"], P["W_E"].T)


def test_data_split_and_windows():
    data, chars = G.load_text()
    assert len(chars) == 65 and sum(len(v) for v in data.values()) == 1115394
    Xw, Yw = G.windows(data["val"], 32)
    assert torch.equal(Xw[:, 1:], Yw[:, :-1]) and torch.equal(Xw[1, 0], Yw[0, -1])


if __name__ == "__main__":
    for n in [n for n in dir() if n.startswith("test_")]:
        globals()[n](); print("ok", n)
