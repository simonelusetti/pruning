# Baseline runners

This directory contains two distinct kinds of comparison. Keep their results separate.

## Published settings

`run_native.py` launches the authors' pinned implementations in their published model and data settings. `bootstrap.py` checks out:

- SparseGPT at `147d2159dc4f3e9f73e47b32c04d7b3708f44436`;
- Wanda at `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`;
- FLAP at `3bb57db3449dd2fa04a5c2192de80e87e33be2b1`.

The default published examples are OPT-125M with C4 calibration for SparseGPT and 7B LLaMA checkpoints for Wanda and FLAP. These runs require their separate environments under `envs/` and normally CUDA.

```bash
python3 native_baselines/bootstrap.py
python3 native_baselines/run_native.py sparsegpt
python3 native_baselines/run_native.py sparsegpt --execute --python "$CONDA_PREFIX/bin/python" --gpu 0
```

The launcher prints commands unless `--execute` is supplied. Native outputs record the source commit, command, model, dataset, seed, and package versions.

## CPU-close OPT comparison

`run_cpu_close.py` ports SparseGPT, Wanda, and FLAP to current PyTorch and Transformers on `facebook/opt-125m`. It is a controlled local comparison, not an official reproduction. The default CPU budget uses shortened calibration and evaluation sequences.

```bash
python3 native_baselines/run_cpu_close.py sparsegpt \
  --nsamples 8 --seqlen 512 --eval-tokens 32768 --threads 16
```

`run_ours_opt.py` applies Classic to the same model. It uses C4 for move statistics, one WikiText-2 split for candidate selection, a disjoint split for the anchored stopping rule, and the same test prefix used by the CPU-close baselines.

```bash
python3 native_baselines/run_ours_opt.py \
  --move-group classic --nsamples 8 --seqlen 512 \
  --val-a 2 --val-b 4 --stats-tokens 1024 --eval-tokens 32768
```

The Classic OPT port uses block proposals for practical runtime. `--move-group classic` runs structural moves; `classic_weights` adds scalar weight zeroing.

`run_opt_fisher_sparsegpt.py` is the maintained scalable Fisher branch for OPT. It follows SparseGPT's layerwise pruning and compensation schedule while replacing its saliency geometry with cached factorized Fisher factors.

```bash
python3 native_baselines/run_opt_fisher_sparsegpt.py --device cuda
python3 native_baselines/run_opt_fisher_sparsegpt.py --device cuda \
  --fisher-geometry axi --fisher-structure none --fisher-decision validate
```

`fisher_sparsegpt` is the default preset: `axb` geometry, OBS deletion, SparseGPT compensation, post hoc canonicalization, and proxy decisions. These axes are stored in every result artifact.

The same runner can use Gurobi for a bounded layer-local Fisher General move:

```bash
python3 native_baselines/run_opt_fisher_sparsegpt.py \
  --fisher-deletion miqp --fisher-compensation sparse_miqp \
  --fisher-update-scope layer --targets .00002 --fisher-pool 128
```

The MIQP pool is a real optimization boundary. A target is rejected when it would require more deletion variables than `fisher_pool`; the runner never reports a small candidate-pool fraction as whole-layer sparsity.

Use `--fisher-repeat --fisher-decision validate --targets .1,.2,.3,.4,.5,.6,.7,.8,.9,1` for a cumulative run stopping at the first loss-limit violation. Each bounded move chooses exactly half a pool of deletions (or the remaining target), refreshes candidate rankings, and permits compensation only in surviving weights of that module. A/B stay fixed within a module; A is collected again at each stage. The reported summed move damage excludes cross-move terms; stopping uses measured loss against the original model. Repeated runs cannot resume without a model checkpoint. Small pools can require hundreds of thousands of solves per stage on OPT-125M.

`--fisher-deletion obs --fisher-compensation complement` fixes a generalized-OBS-ranked deletion mask and optimizes every surviving weight in the same linear module. It solves the constrained AXB quadratic plus `fisher_mu / 2` times the squared surviving update norm using matrix-free preconditioned conjugate gradients. Both factors retain off-diagonal interactions; the full Kronecker matrix is never constructed. The spectral preconditioner uses the unrestricted regularized Kronecker inverse. The solver checks the true relative residual and fails if it exceeds `--fisher-cg-rtol` after `--fisher-cg-max-iterations`; deleted weights remain exactly zero. There is no update support penalty or compensation bound. Deletion selection remains a ranking approximation rather than joint support optimization. Combine with `--fisher-repeat` for cumulative targets; `--fisher-structure none` leaves the deletion mask unaltered by post hoc canonicalization.

Parameter removal is reported against all OPT parameters. The moves affect decoder linear weights and biases, so the maximum reachable fraction is smaller than 100%. Dense tensor shapes are retained; counts represent effective parameters and do not imply checkpoint-size or latency savings.

The maintained modular-addition SparseGPT port writes only its result artifact; plotting is handled separately:

```bash
python3 native_baselines/run_modadd_sparsegpt.py
```

## Berman comparison

`run_berman_mnist.py` compares the official released SimpleNN/MNIST checkpoint with Berman, Classic S, Classic W+S, and Fisher General. Candidate choice uses `valA`; the anchored `2 SE + rho` stopping rule uses disjoint `valB`; test loss is only reported.

`python3 native_baselines/run_berman_mnist.py --method fisher_sparsegpt --device cpu` runs Fisher-aware SparseGPT on the same checkpoint and splits. It freezes class-marginal downstream B factors estimated from 500 training images, refreshes activation A before each linear module, and uses the OPT branch's generalized-OBS ranking and later-column compensation. Each 10% grid target starts from the original checkpoint; biases stay fixed before post hoc canonicalization. It stops at the first validation-limit violation, writes `results_berman/simplenn_mnist/fisher_sparsegpt.json`, and updates the Berman comparison plot.

`run_compensation_ablation.py --setting {berman,opt} --target .3 --output PATH` compares deletion alone with compensation scales 0, .25, .5, .75, and 1 on a single fixed original-model AXB-OBS mask. It freezes all factors at the original model, fixes biases, disables canonicalization, and checks the regularized quadratic against deletion alone. Berman additionally solves the exact class-marginal Fisher system across all surviving layer weights through Fisher-vector products. Metrics include paired validation/held-out cross-entropy changes and output KL; calibration KL is also evaluated. OPT uses the published 128x2048 C4 calibration pool, the existing four-sequence selection split, and eight independently seeded held-out sequences. Berman uses the existing 500-image Fisher subset and disjoint validation/test splits. The default ridge is 1e-4 and CG relative residual tolerance is 1e-6. `plot_compensation_ablation.py` plots the saved results separately from sparsity sweeps.

```bash
python3 native_baselines/run_berman_mnist.py --method all --rho 0.05 --n-sigma 2
```

The released checkpoint is one trained seed. A paper-style mean curve requires independently trained checkpoints and must not be approximated by smoothing this trajectory.

## Plots

`plot_comparisons.py` reads saved JSON artifacts and writes the combined figures under `plots/`. It checks that parameter removal is monotone and that local runs share the same checkpoint and stopping protocol. Invalid or incompatible results fail loudly instead of being silently combined.
