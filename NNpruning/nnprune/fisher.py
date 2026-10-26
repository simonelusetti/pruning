from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FisherStrategy:
    geometry: str
    deletion: str
    compensation: str
    update_scope: str
    structure: str
    decision: str

    def as_dict(self):
        return asdict(self)


PRESETS = {
    "fisher_general": FisherStrategy(
        geometry="full", deletion="miqp", compensation="sparse_miqp",
        update_scope="global", structure="reward", decision="validate"),
    "fisher_sparsegpt": FisherStrategy(
        geometry="axb", deletion="obs", compensation="sparsegpt",
        update_scope="layer", structure="post", decision="proxy"),
}


def resolve_strategy(preset: str, **overrides) -> FisherStrategy:
    try:
        strategy = PRESETS[preset]
    except KeyError as error:
        raise ValueError(f"unknown Fisher preset: {preset}") from error
    return replace(strategy, **{key: value for key, value in overrides.items()
                                if value is not None})


def validate_strategy(strategy: FisherStrategy, backend: str) -> FisherStrategy:
    supported = {
        "mlp": {
            "geometry": {"full"},
            "deletion": {"miqp"},
            "compensation": {"sparse_miqp", "dense"},
            "update_scope": {"global"},
            "structure": {"reward", "post"},
            "decision": {"validate", "proxy"},
        },
        "opt": {
            "geometry": {"axb", "axi"},
            "deletion": {"obs", "miqp"},
            "compensation": {"sparsegpt", "sparse_miqp", "dense", "complement"},
            "update_scope": {"layer"},
            "structure": {"post", "none"},
            "decision": {"validate", "proxy"},
        },
    }
    if backend not in supported:
        raise ValueError(f"unknown Fisher backend: {backend}")
    for field, choices in supported[backend].items():
        value = getattr(strategy, field)
        if value not in choices:
            options = ", ".join(sorted(choices))
            raise ValueError(f"{backend} does not support Fisher {field}={value}; "
                             f"choose {options}")
    if backend == "opt":
        valid_compensation = ({"sparsegpt", "complement"} if strategy.deletion == "obs" else
                              {"sparse_miqp", "dense"})
        if strategy.compensation not in valid_compensation:
            raise ValueError(
                f"OPT deletion={strategy.deletion} is incompatible with "
                f"compensation={strategy.compensation}")
        if strategy.deletion == "miqp" and strategy.geometry != "axb":
            raise ValueError("OPT layer-local MIQP currently requires axb geometry")
        if strategy.compensation == "complement" and strategy.geometry != "axb":
            raise ValueError("OPT full-complement compensation requires axb geometry")
    return strategy
