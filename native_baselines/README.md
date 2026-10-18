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

`run_ours_opt.py` applies the framework to the same model. It uses C4 for move statistics, one WikiText-2 split for candidate selection, a disjoint split for the anchored stopping rule, and the same test prefix used by the CPU-close baselines.

```bash
python3 native_baselines/run_ours_opt.py \
  --move-group classic --nsamples 8 --seqlen 512 \
  --val-a 2 --val-b 4 --stats-tokens 1024 --eval-tokens 32768
```

The classic OPT port uses block proposals for practical runtime. The Fisher OPT port uses exact compensated single or pair deletions after a rough layer-local shortlist; it therefore must be labelled as screened Fisher. It does not enumerate every OPT neuron or head. Candidate-budget fields in every result record this difference.

Parameter removal is reported against all OPT parameters. The moves affect decoder linear weights and biases, so the maximum reachable fraction is smaller than 100%. Dense tensor shapes are retained; counts represent effective parameters and do not imply checkpoint-size or latency savings.

The maintained modular-addition SparseGPT port writes only its result artifact; plotting is handled separately:

```bash
python3 native_baselines/run_modadd_sparsegpt.py
```

## Berman comparison

`run_berman_mnist.py` compares the official released SimpleNN/MNIST checkpoint with classic structural moves. Candidate choice uses `valA`; the anchored `2 SE + rho` stopping rule uses disjoint `valB`; test loss is only reported.

```bash
python3 native_baselines/run_berman_mnist.py --method all --rho 0.05 --n-sigma 2
```

The released checkpoint is one trained seed. A paper-style mean curve requires independently trained checkpoints and must not be approximated by smoothing this trajectory.

## Plots

`plot_comparisons.py` reads saved JSON artifacts and writes the combined figures under `plots/`. It checks that parameter removal is monotone and that local runs share the same checkpoint and stopping protocol. Invalid or incompatible results fail loudly instead of being silently combined.
