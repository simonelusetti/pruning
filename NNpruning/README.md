# NNpruning

The project has three maintained pruning paths:

- `classic`: the supervisor's structural moves. Set `--weights` to obtain W+S; omit it for S only.
- `fisher_general`: graph-aware parameter deletion and compensation solved by Gurobi MIQP. It currently supports Bernoulli MLPs.
- Fisher-aware SparseGPT: the scalable OPT path in `native_baselines/run_opt_fisher_sparsegpt.py`.

Classic remains its own algorithm. The Fisher paths are presets over five explicit axes:

| preset | geometry | deletion | compensation | update scope | structure | decision |
| --- | --- | --- | --- | --- | --- | --- |
| `fisher_general` | full Fisher on the bounded MLP pool | MIQP | sparse MIQP | global | reward canonical closure | validate |
| `fisher_sparsegpt` | cached `A x B` | generalized OBS | SparseGPT propagation | layer | post-process | proxy |

The MLP backend also supports dense compensation, post-process structural accounting, and proxy decisions. The OPT backend supports `A x I`, disabling post-process canonicalization, target-wise validation, and a bounded layer-local MIQP with `A x B`. Unsupported combinations fail before a run starts. Two-moons is the quick MLP benchmark; modular addition is the minute-scale transformer benchmark and supports Classic only. Berman, SparseGPT, Wanda, and FLAP remain comparison baselines rather than framework branches.

```bash
cd NNpruning
python3 run_experiments.py moons --moves classic
python3 run_experiments.py moons --moves classic --weights
python3 run_experiments.py moons --moves fisher_general
python3 run_experiments.py moons --moves fisher \
  --fisher-preset fisher_general --fisher-compensation dense \
  --fisher-structure post --fisher-decision proxy
python3 run_experiments.py modadd --moves classic --device cuda
```

Adapters and states move between CPU and CUDA with `.to(device)`.
