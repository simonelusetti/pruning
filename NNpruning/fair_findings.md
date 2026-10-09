These findings are for the cached run: MLP seeds 123, 7 and 42, transformer seed 0.

**1. Held-out data in the loop is not where the advantage comes from.** In both
models, choosing each move on training data gives the same result as choosing
it on valA.

| model | arm | train-selected | valA-selected |
|---|---|---|---|
| MLP | W+S | 94% | 97% |
| MLP | S | 94% | 95% |
| transformer | W+S | 65% | 66% |

(Margin 1% for the MLP, margin = base loss for the transformer.)

**2. Composite / structural moves are where it comes from.** With the same
proposer (Berman et al.'s score), the same tie-break and the same stop rule:

- **MLP, margin 1%:**

  | arm | % removed |
  |---|---|
  | single parameters, greedy | 10–21 |
  | single parameters, one-shot (Berman's family) | 6–9 |
  | structural, greedy | 94–97 |

  At matched sparsity, structural arms keep the base test loss (0.132) up to
  90% removed. The best single-parameter arm is already at 0.31 at 50%.
- **Transformer:**

  | arm | test accuracy at 70% removed |
  |---|---|
  | structural | 1.000 |
  | single parameters, greedy | 0.47–0.50 |
  | Berman iterative, one-shot | 0.10 |
  | magnitude | 0.06 |

  Here the combination W+S beats S alone. S on its own runs out at 86%
  (embeddings and unembedding have no structural moves) and is less accurate
  at 70% (0.84–0.96).

**3. Greedy loss evaluation helps single parameters, but only moderately.** The
gain is MLP 6% → 10–21%, and on the transformer test accuracy at 50% goes from
0.68 to 0.999. One-shot structural pruning (classical mean-ablation) stays
poor: 14% on the MLP, 13% on the transformer. Composite moves pay off only when
each one is evaluated.

**4. The pure 2σ rule needs a practical margin.** With δ = 0 the MLP stop points
are bimodal (sd ≈ 47 points across seeds). A move that changes the loss by
0.0002 with SE 0.0001 is "significant" and ends the path, even though later
steps pass again. Any margin ≥ 0.5% of the base loss removes this. On the
transformer (base loss 8·10⁻⁴), δ = 0 and δ = 1% give the same stops.

**Caveats.**
- One transformer seed.
- Single-parameter moves exclude biases; Berman's protocol prunes them too.
- Structural moves carry their own closed-form compensation (folds, merges),
  which is part of what "composite" means here. Ablating that compensation is
  the next test.
- Counts are parameters (k_eff), not FLOPs.
