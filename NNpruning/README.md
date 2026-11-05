# NNpruning

This directory contains the algorithm and benchmark library. Run maintained experiments through Forge from the repository root:

```bash
forge run benchmark=moons branch=classic
forge run benchmark=moons branch=fisher_general
forge run benchmark=modadd branch=classic runtime.device=cuda:0
```

The supported local benchmarks are two-moons for quick checks and modular addition for longer checks. The maintained algorithms are Classic structural pruning, Classic with weight zeroing, and Fisher General MIQP on Bernoulli MLPs. The OPT Fisher-aware SparseGPT implementation is in `pruning_experiments/opt_fisher.py` and is also selected through Forge. Configuration details are in the root README and `conf/`.
