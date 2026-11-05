# Pruning project: methods and current evidence

## Research question and common formulation

Our Fisher-based pruning methods describe a move using two sets: $D$, parameters fixed to a constant (currently zero), and $U$, surviving parameters allowed to change by an amount $u$. Their combined change is $\Delta(D,u)$.

We need a cheap way to judge possible moves before measuring their actual loss. We use a quadratic approximation around the current model, represented by a positive-semidefinite matrix $F$. Choosing that matrix is itself a modeling decision: it may be a full-model Fisher estimate or a cheaper local approximation. Loss validation remains part of the procedure, determining whether a proposed move is accepted; it does not become unnecessary because we use a proxy.

The family of objectives we study is

$$
\mathcal J(D,u)=\tfrac12\Delta(D,u)^\top F\Delta(D,u)
+\tfrac{\mu}{2}\|u\|_2^2
+\lambda_u|\operatorname{supp}(u)|
-\lambda_d K(D).
$$

The terms estimate damage from the whole move, discourage large updates, charge for updating many weights, and reward structural savings $K(D)$, respectively. Canonicalization can remove connections made redundant by a deletion while preserving the represented function, under the conditions handled by the canonicalizer. Individual branches may set some coefficients to zero; Classic's structural moves are separate baselines.

For a **fixed** $D$, if all eligible survivors may change and the update-count penalty is absent, the best update for this quadratic solves

$$
(F_{UU}+\mu I)u=-F_{UD}d,\qquad d=-\theta_D.
$$

Here $K(D)$ is already fixed, so it does not obstruct this update solve. The difficult part is selecting $D$: that remains discrete even when we omit canonicalization from the search.

We use two operational modes:

- **Rank, then solve.** Score each possible single deletion using the Optimal Brain Surgeon (OBS) formula $s_i=\theta_i^2/[2(F^{-1})_{ii}]$ for an invertible or damped $F$. It allows all other eligible weights to compensate in that *one-deletion* calculation. Take the lowest scores to form $D$, then solve the quadratic for its surviving weights. The update is the conditional optimum up to numerical tolerance; combining many individually attractive deletions is a heuristic, since their interactions were not considered while ranking.
- **Joint support search.** On the smaller MLP, a Gurobi mixed-integer quadratic program chooses deletion blocks and a limited set of compensating coordinates together and rewards canonicalized parameter savings. Its candidate blocks are ranked before solving, so it searches a bounded pool rather than every possible model change. This is much more expensive.

## Current results

**Berman SimpleNN/MNIST.** Fisher General uses a full-model empirical Fisher. Its MIQP jointly chooses from ranked batches of deletions and a small update pool, and rewards extra parameters removed by canonicalization. Berman's iterative baseline ranks individual parameters by a diagonal Fisher score, prunes in 10-percentage-point increments, and does not compensate surviving parameters. The table reports **held-out test-loss change relative to the original model**; negative values mean lower loss. Values are rounded readings at these removal levels from the [retained comparison](../plots/berman/comparison.png), not exact saved measurements.

| Parameters removed | Berman iterative | Fisher General MIQP |
|---:|---:|---:|
| 0% | 0% | 0% |
| about 10% | about 0% | about 0% |
| about 20% | −0.3% | −0.1% |
| about 30% | −2.1% | −1.1% |
| about 40% | −3.1% | −2.9% |
| about 45% | +0.9% | −4.7% |
| about 50% | +5.2% | −3.7% |

The clearest advantage is near **45–50% removal**. The two curves are close at lower sparsities; Fisher General's advantage is not uniform across every step.

**OPT-125M.** Fisher-aware SparseGPT uses $A\otimes B$ to score deletions within input-column blocks, then mainly uses SparseGPT's sequential $A$-based update. The newer full-complement branch ranks deletions once per linear module and lets *all* surviving module weights compensate under a local $A\otimes B$ quadratic. Its $B$ was refreshed from 16,256 sampled gradient observations before each transformer block. The table gives **C4 selection-loss increase relative to the original model**. The SparseGPT values are approximate readings near each removal level from the [retained comparison](../plots/opt/comparison_legacy.png); the full-complement values are exact from the completed Forge run.

| Decoder-linear parameters removed | Native SparseGPT | Fisher-aware SparseGPT | Refreshed-$B$ full complement |
|---:|---:|---:|---:|
| 0% | 0% | 0% | 0% |
| about 10% | about +0.3% | about +0.2% | +0.234% at 9.99% |
| about 20% | about +0.6% | about +0.5% | +0.591% at 19.98% |
| about 30% | about +1.4% | about +1.2% | +1.784% at 29.97% |
| about 40% | about +3.1% | about +2.5% | +6.759% at 39.96%; rejected |
| about 45% | about +4.6%; stopped | about +3.9% | — |
| about 50% | — | about +6.7% | — |

The full-complement run used **four C4 selection sequences** and stopped after the rejected 40% proposal. The older SparseGPT curves used different evaluation offsets and differ in achieved removal and other settings, so their apparent ordering is not yet a matched baseline comparison. A separate 10% diagnostic on 128 held-out sequences found 0.215% loss increase for full complement and 0.255% for a Fisher-aware SparseGPT reference, but those methods removed 9.99% and 11.05% of parameters. Two sequential $A\otimes B$ variants in that diagnostic were better still (0.189% and 0.191%).

## Interpretation and next question

The project has one Fisher-based objective but **two different promising implementations**: bounded MIQP with a full-model Fisher and structural reward on Berman, and a sequential local approximation on OPT. That is a coherent methodological family, but it is not yet evidence that one algorithm beats both baselines. The current question is whether the OPT full-complement shortfall comes mainly from its one-pass deletion mask or from the local quadratic being a poor predictor of real loss after broad updates. A matched-mask diagnostic is configured to separate those effects. The [implementation reference](implementation_reference.md) records the exact algorithms, settings, and comparison limits for further analysis.
