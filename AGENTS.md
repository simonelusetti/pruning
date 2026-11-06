# Repository rules

- Keep the framework small. Add comments only when they explain a non-obvious invariant needed for maintenance.
- Do not save experiment tables, logs, reports, or notes unless the user explicitly asks for an artifact.
- Keep tensor work in PyTorch and batch data with `torch.utils.data.DataLoader`; use Python loops only for structurally different candidates or batches.
- New code must run on CPU or CUDA through an explicit device, without device-specific branches in the algorithms.
- Maintained methods are `classic` (S or W+S), `fisher_general` (Gurobi MIQP on Bernoulli MLPs), and Fisher-aware SparseGPT for OPT.
- The maintained local benchmarks are two-moons for quick checks and modular addition for minute-scale checks. Native SparseGPT, Wanda, and FLAP scripts retain their published model/data settings.
- Do not add CNN or Fashion-MNIST code.
- Launch long-running experiments in a detached `tmux` session and verify the worker remains active after launch.
