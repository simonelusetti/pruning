# Handoff: theory of structure-aware second-order pruning

You are continuing the theoretical side of a research project on pruning neural networks. The code will be linked
separately; it is the source of truth for what is implemented. This document records what the code cannot tell you:
the spirit of the project, the vocabulary, the reasoning behind design decisions, the related work we considered (and
how firmly each claim is established), the conclusions reached so far, and the open questions. State of 2026-11-15.

## Working style the user expects

- Be concise; give the gist first. Equations render as LaTeX (`$...$`, `$$...$$`) and are preferred to Unicode.
- Prefer principled, parameter-free formulations. A free trade-off coefficient (a λ with units) is a smell; prefer
  constrained forms or coefficients fixed by theory (e.g. confidence bounds). The user assumes a cleaner, more general
  formulation is also more likely to be correct.
- Always distinguish (i) published claims, (ii) project notes/derivations, (iii) your own new reasoning. Several
  references below came from abstracts or memory only; say so when you rely on them. Papers the project builds on
  should live in `sources/` (the repo's reference archive).
- Correct the user directly when a premise is wrong (e.g. "isn't this just X?" → explain the actual difference).

## Spirit of the project

The project began from the supervisor's *boundary pruning* framework for symbolic regression (sources: `Emergent_SR`,
`NNpruning — working notes`, original code). Its ideas, carried to neural networks:

- Every pruning move is a **limit of the parametrisation, realised exactly**: zeroing a weight, saturating a unit
  (bias → ±∞), replacing a unit by its mean (folded into the next bias), merging duplicates, rank cuts. Moves follow
  from the architecture rather than a list of heuristics.
- **Identifiability decides how a move is judged.** *Exact* redundancy (a structural null) is removed at zero cost
  with no statistic; *sloppy* redundancy has no valid Fisher statistic and needs held-out data.
- The `classic` branch is this framework (greedy structural moves, held-out selection, BIC-style tie-break).

The user's expansion is a **second-order, joint** view. Choose a deletion set D and compensating values u on surviving
coordinates U to minimise the Fisher quadratic damage, while crediting the **structural savings K(D)**: everything
that function-preserving cleanup (*canonicalization*) frees once D is deleted — dead neurons (all incoming or all
outgoing weights gone), constants folded into downstream biases, cascades. Historical objective:

$$
\tfrac12\,\Delta(D,u)^\top F\,\Delta(D,u) \;+\; \tfrac{\mu}{2}\|u\|^2 \;+\; \lambda_u\,|\mathrm{supp}(u)| \;-\; \lambda_d\,K(D).
$$

**The current research focus is operationalising "exploit the model's structure" through K(D)**, efficiently and with
theory behind it. Three axes that used to be conflated under "joint": (1) whether/how U is charged, (2) including K(D)
(which requires optimising over *sets*), (3) the algorithm (until recently only Gurobi).

## Vocabulary and settled facts

- **F** is the *model* Fisher (labels sampled from the model). $\tfrac12\delta^\top F\delta$ approximates the average KL
  between dense and pruned predictions; it is ≥ 0 and zero only at δ = 0, so this objective can never say pruning
  *improves* the model. The per-example KL has **no linear term** at δ = 0 for any weights (the score has zero mean
  under the model's own distribution); its Hessian is the per-example Fisher $F_i$.
- **K-FAC**: block-diagonal per affine module, $F_m \approx B_m\otimes A_m$ (A: input second moments incl. a bias
  column; B: sampled output-gradient second moments). Consequences: the optimal compensation for deletions in module m
  lies entirely in m; with a fixed per-module budget, deletion selection decomposes per module exactly.
- **Compensation** = solving the AXB quadratic over all surviving coefficients of a module ("full complement",
  conjugate gradient). **OBS singleton score** $\tfrac12 w^2/[F^{-1}]_{ii}$ is the cost *with* compensation; the OBD
  score $\tfrac12 w^2 F_{ii}$ is the cost *without*.
- **Shared settings** (meaning fixed across methods):
  - `scope: local | global`. Local = each solve unit meets the target alone (quota). Global = one model-wide problem and budget. Under K-FAC, global reduces to per-unit problems linked by one shared price per deletion, so global vs local differs only in budget sharing. With the exact Fisher, global also couples modules.
  - `replay` (local only): prune block by block, re-measuring statistics on the already pruned model (incompatible with a single global budget).
  - `unit: weight | neuron | both`.
  - `affine`: biases are candidates and compensation variables, or frozen.
  - `curvature: kfac | fisher`.
  - `objective: kl | spread`.
  - `compensation: full | none`.
- **Methods** (branches):
  - `singular`: score candidates once by singleton OBS cost per removed parameter, select up to budget, then one full compensation per module.
  - `joint`: per greedy move, exact MIQP (Gurobi) over a tiny pool (~8 blocks + 8 update coordinates) with an update charge and exact K via logical constraints; everything recomputed per move.
  - `joint_simple`: canonical L0 by penalised proximal gradient on $\tfrac12(V-W)^\top F(V-W)+\lambda\,\mathrm{retained}(V)$ over all coefficients. K enters through a three-state neuron prox (keep / constant / disconnected); λ is searched per step; no update charge.
  - `joint_submodular`: greedy cover over lifted candidates (below), with neuron cuts applied one at a time with exact rank-one compensation, weights in rounds, and a full solve per round.
  - `classic`: the supervisor's framework.
  - Baselines: Berman et al. (Fisher-distance scores), SparseGPT, Wanda, FLAP, SliceGPT and LLM-Pruner (authors' code), and a magnitude/random reference.
- **Protocol**: every method prunes progressively until held-out validation loss exceeds the dense loss by 5% + 2
  paired standard errors; fixed-target methods sweep 5% steps from dense. Benchmarks: two-moons, modular addition,
  "berman" (MNIST SimpleNN from Berman et al.), OPT-125M (SparseGPT setting), OPT-125M SliceGPT setting, LLaMA-7B
  settings (blocked by GPU memory).

## Theoretical conclusions reached so far

1. **K(D) is not submodular in weight space.** Deleting the last live weight of a neuron frees the whole neuron
   (increasing returns). Example: incoming a, b, outgoing c: $K(\{b\})-K(\varnothing)=1<K(\{a,b\})-K(\{a\})=2$. The
   "all incoming OR all outgoing" death rule means it is not cleanly supermodular either.
2. **Lifting restores submodularity exactly.** Add, per hidden neuron, two candidate *copies*: a fan-in cut (the neuron
   becomes a constant folded downstream) and a fan-out cut. Each candidate covers coordinates (a weight itself; a neuron
   copy its row, column and own bias). Savings are then a **coverage function** $K(T)=|\bigcup_{v\in T}\mathrm{cover}(v)|$,
   which is submodular, and still exact: the increasing returns were an artifact of describing structures only through
   their weights. Both cuts cover the same coordinates; they differ only in compensation (the fan-in cut alters the
   protected downstream bias). After a cut, the rest of the neuron is frozen (it neither compensates nor is refit; cleanup
   removes it), in every method. Gap: multi-level cascades in chains of ≥ 3 layers. Exact for 2-layer MLP blocks (every
   OPT/LLaMA MLP); NOT exact on the berman MNIST model (784-64-64-10 is a 3-layer chain).
3. **Use the constrained form.** K counts parameters and damage is in nats, so $K-\mathrm{damage}$ fixes an arbitrary
   exchange rate. Prefer $\min\ \mathrm{damage}\ \text{s.t.}\ K\ge k$ (progressive targets) or
   $\max K\ \text{s.t.}\ \mathrm{damage}\le\rho$; λ becomes a Lagrange multiplier. Iyer & Bilmes also show the
   difference form is inapproximable in the worst case.
4. **Damage is weakly submodular.** Compensated damage is a best-subset-type objective; restricted strong convexity
   implies weak submodularity (Elenberg et al.), with ratio set by the conditioning of F on sparse sets. Damping improves
   it. This is what gives greedy over lifted candidates its footing.
5. **Incremental updates under K-FAC.** Removing a row (or column) of module m is a Schur-complement update of the
   restricted $B^{-1}$ (or $A^{-1}$), O(d²), exact; scattered single weights break the Kronecker structure (no closed
   form) and must be batched with a CG solve between batches.
6. **The spread objective (implemented).** Use the per-example KL from the dense model for both terms (no labels, no
   linear terms):
   $$\min_{\delta:\ \delta_D=-w_D}\ \mathbb{E}_i[\mathrm{KL}_i]+\beta\sqrt{\mathrm{Var}_i[\mathrm{KL}_i]},\qquad \beta=\sqrt{2\log(1/\eta)/n}.$$
   To leading order $\mathrm{Var}_i[\mathrm{KL}_i]=\tfrac14\mathrm{Var}_i[\delta^\top F_i\delta]$ (quartic; spread of squared
   prediction distortion $\|J_i\delta\|^2_{H_i}$). Its gradient is a reweighted Fisher, so it is optimised by IRLS:
   $F_w=\mathbb{E}_i[(1+\beta z_i)F_i]$ with $z_i$ the standardised exact per-example KL of the current model. The user
   preferred this bound form over mean + λ·variance because the bound's coefficient is dimensionless and fixed by n, η.
7. **Compensation is a dial between generalisation and robustness** (empirical, berman). Uncompensated deletion acts
   as shrinkage on an overfit model: held-out loss drops ~5% below dense around 40–46% removed, then collapses (~45–58%).
   Full compensation reproduces the (overfit) dense model, so no dip, but survives to ~80%. The MIQP's dip came from its
   tiny update budget, not Gurobi: 0 / 8 / 64 updates give progressively shallower dips and later breaks; charging updates
   (λ_u) is irrelevant. On OPT-125M no method dips (the pretrained model is not overfit to its evaluation data). Using the
   OBS score without compensation is mis-specified; uncompensated variants now rank by the OBD cost (the earlier
   uncompensated curves used OBS and are being rerun).

## Related work considered

Status markers: [S] PDF in `sources/` (read in full or substantially); [A] found by web search, abstract/summary only;
[M] recalled from memory, not checked this project.

Combinatorial / second-order pruning:
- [S] CHITA (Benbaki et al., ICML 2023): OBS with Fisher, combinatorial, unstructured, Hessian-free IHT.
- [A] OSSCAR (Meng et al., ICML 2024): one-shot structured pruning as combinatorial optimisation over subsets of
  structures, local search with low-rank updates, scales to OPT-30B.
- [A] FALCON (Mazumder group, 2024): ILP jointly under FLOP and sparsity constraints.
- [A] ALPS (Meng et al., 2024): L0-constrained layer-wise pruning solved with ADMM plus PCG refit; strong at 70%.
- [A] iCBS (Rosenberg et al., Amazon, 2024): block-coordinate combinatorial brain surgeon (QCBO per block). Most direct
  way to scale an exact MIQP.
- [A] ZipLM (Kurtic et al., NeurIPS 2023): structured OBS, inference-aware greedy removal.
- [A] Knapsack pruning (Aflalo et al., 2020): neurons as knapsack items.
- [A] DepGraph (Fang et al., CVPR 2023): automatic coupled-parameter groups from the graph (generic version of our
  inferred canonical structure).
- [A] Hazimeh, Mazumder & Radchenko (Ann. Stat. 2023): group L0 with local search and custom branch-and-bound to 10⁶
  features. [A] Bertsimas & Van Parys (Ann. Stat. 2020): exact sparse regression via cutting planes at 10⁵ scale.
- [S] L0Learn (Hazimeh & Mazumder): penalised proximal gradient for L0 (basis of `joint_simple`).
- [A] Louizos et al. (ICLR 2018): hard-concrete L0 gates, also group gates.

Set functions:
- [A] El Halabi, Srinivas & Lacoste-Julien (NeurIPS 2022): neuron selection with compensation is weakly submodular;
  greedy with guarantees.
- [A] Iyer & Bilmes (UAI 2012): difference of submodular functions (MM algorithms); [A] Iyer & Bilmes (NeurIPS 2013):
  submodular cover / submodular knapsack constraints (iterate modular bounds) — closest to our constrained form.
- [A] Das & Kempe (ICML 2011) submodularity ratio; [A] Elenberg et al. (2016) RSC ⇒ weak submodularity.
- [A] Harshaw et al. (ICML 2019): weakly submodular minus modular cost (distorted greedy).
- [A] Bai & Bilmes (ICML 2018): submodular + supermodular (curvature-dependent greedy).
- [M] Nemhauser–Wolsey–Fisher, Feige (max coverage), Minoux lazy greedy, Mirzasoleiman stochastic greedy, Wolsey
  submodular cover, Buchbinder double greedy.

Variance and generalisation:
- [M] Maurer & Pontil (2009) empirical Bernstein bounds; [M] Namkoong & Duchi (NeurIPS 2017) variance regularisation ≈
  χ²-DRO. Basis of the β in the spread objective; verify before citing.
- [M] Heskes (1998) bias–variance for log loss; [M] Molchanov et al. (2017) variational dropout sparsification; [M]
  Louizos et al. (2017) Bayesian compression; [M] Achille & Soatto, information in the weights.

The supervisor's three manuscripts ([S], in `sources/`):
- Achille & Soatto, *Information Dropout* (1611.01353): Information Bottleneck via learned multiplicative noise per
  unit; with ReLU and a log-uniform prior each unit costs $-\log\alpha_j$; a unit tolerating unbounded noise carries no
  information. Bridge (ours): post-training, noise on unit j costs $\tfrac12\alpha_j c_j$ (our uncompensated column
  cost); $\max\sum\log\alpha_j$ s.t. budget gives $\alpha_j\propto1/c_j$; differs from our ranking by dropping the
  per-parameter normalisation (cost vs cost/size) — a candidate "K-free" structural reward.
- Yu et al., *Effective Information Criterion* (yu26az, ICML 2026): multiplicative rounding noise per node,
  $s_k^2=1+\sum_i\kappa_{ki}^2 s_i^2$, EIC = worst node; judges the computation graph, not the function. Candidate
  "error-propagation safety" diagnostic for pruned networks.
- Moscato, *Sequential Newton–MIMEP* (manuscript_final): representation depth vs scientific depth; admit one
  coordinate at a time; protected tails in the null space of admitted quantities; profiled information
  $I_{x|N}$; nuisance monotonicity (enlarging free nuisance directions cannot increase profiled information); inherited
  error amplification $g^\top\Sigma g$. Links: OBS score = Wald statistic with all other coefficients as nuisance;
  compensation = nuisance refit; our validation stop = admission test.

## Supervisor's open requests (her words, paraphrased)

- Pruning as **variance reduction and error-propagation safety**; hope: a variance objective might make an explicit
  K(D) unnecessary. Our assessment so far: noise-site choice plays the role of K; variance objectives are soft (exact
  zeros only for exact nulls); log-scale (IB-type) rewards, not raw variance, can stand in for a count.
- "Out of curiosity/validation, look at the **evolution of the bias–variance trade-off** if computationally viable."
  Not yet done. Distinguish variance *across examples* (what the spread objective uses) from variance *across training
  sets* (the bias–variance notion). Options: retrain replicates on small benchmarks (Heskes decomposition), or Laplace
  predictive variance $J F^{-1}J^\top/n$ on surviving parameters. Held-out loss variance per step is already logged.

## Open questions

- Guarantees for greedy cover with exact lifted K plus compensated damage (coverage reward, weakly submodular cost,
  constrained/cover form); role of the multi-level cascade gap.
- A principled partial-compensation axis (the generalisation/robustness dial) and choosing the operating point by
  validation rather than at the break point.
- Whether a K-free, information/variance-based structural reward (Information Dropout bridge) matches K(D) on the
  parameter axis, or only on robustness axes.
- Whether spread helps at all: the first spread runs weighted the bias column of A incorrectly (fixed); the
  observation "spread helps `joint_simple` (break 65% → 75%) but not `joint_submodular`" awaits the rerun.
