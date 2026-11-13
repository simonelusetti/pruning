from __future__ import annotations
import hashlib
import torch
from pruning.nnprune import mlp as M, transformer as T
from pruning.paths import ROOT
CHECKPOINTS = ROOT / "outputs/checkpoints"
CHECKPOINT_NAMES = {"moons": "moons_seed123.pt", "modadd": "modadd_seed0.pt"}


def split_validation(data: dict, seed: int = 0) -> dict:
    out = dict(data)
    X, y = data["val"]
    if len(X) < 2:
        raise ValueError("validation data needs at least two examples")
    generator = torch.Generator(device=X.device).manual_seed(seed + 1000)
    order = torch.randperm(len(X), generator=generator, device=X.device)
    half = len(X) // 2
    out["valA"] = X[order[:half]], y[order[:half]]
    out["valB"] = X[order[half:]], y[order[half:]]
    return out


def load_benchmark(name: str, device="cpu"):
    device = torch.device(device)
    if name == "moons":
        data, _ = M.make_moons_data(seed=123)
        adapter = M.MLPAdapter(data).to(device)
        checkpoint = CHECKPOINTS / CHECKPOINT_NAMES[name]
        train = lambda: M.train_mlp(adapter, seed=123, verbose=False)[0]
    elif name == "modadd":
        data = T.make_modadd_data(seed=0)
        adapter = T.TransformerAdapter(data, p=59).to(device)
        checkpoint = CHECKPOINTS / CHECKPOINT_NAMES[name]
        train = lambda: T.train_transformer(adapter, steps=3000, seed=0,
                                             verbose=False)[0]
    else:
        raise ValueError(f"unknown benchmark: {name}")
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False).to(device)
    else:
        state = train()
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save(state.to("cpu"), checkpoint)
    return adapter, state


def checkpoint_sha256(name: str) -> str:
    path = CHECKPOINTS / CHECKPOINT_NAMES[name]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalProblem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.adapter, self.state = load_benchmark(cfg.benchmark.name, cfg.runtime.device)
        self.adapter.data = split_validation(self.adapter.data, cfg.benchmark.validation_split_seed)
        self.total = self.adapter.k_eff(self.state)
        self.canonical_evaluation = cfg.branch.name in {"berman", "reference"}
        self._selection_reference = None

    def evaluation_state(self):
        if not self.canonical_evaluation:
            return self.state
        state = self.state.clone()
        self.adapter.canonicalize(state)
        return state

    def remaining(self):
        return self.adapter.k_eff(self.evaluation_state())

    def model_parameters(self):
        return self.total

    def parameter_bytes(self):
        return self.remaining() * next(iter(self.state.params.values())).element_size()

    def outside_removed(self):
        return 0

    def compact(self):
        pass

    def profile(self):
        from pruning.models.base import profile
        return profile(self)

    def snapshot(self):
        return self.state.clone()

    def restore(self, state):
        self.state = state.clone()

    def losses(self, split):
        name = {"selection": "valA", "validation": "valB", "reporting": "test"}[split]
        return self.adapter.per_sample(self.evaluation_state(), name, "loss").double()

    def selection_reference(self):
        if self._selection_reference is None:
            self._selection_reference = self.losses("selection")
        return self._selection_reference

    def split_metadata(self, split):
        name = {"selection": "valA", "validation": "valB", "reporting": "test"}[split]
        return {"dataset": self.cfg.benchmark.name, "split": name,
                "count": len(self.adapter.data[name][0]),
                "seed": self.cfg.benchmark.validation_split_seed}

    def prepare(self, index):
        return {}

    def fisher_data(self, count, seed):
        from pruning.models.base import fisher_subset
        return fisher_subset(self.adapter.data["train"], count, seed, self.state.params[next(iter(self.state.params))].device)
