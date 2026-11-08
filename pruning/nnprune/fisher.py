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
    "discrete": FisherStrategy(
        geometry="full", deletion="miqp", compensation="sparse_miqp",
        update_scope="global", structure="reward", decision="validate"),
    "continuous": FisherStrategy(
        geometry="axb", deletion="obs", compensation="complement",
        update_scope="layer", structure="none", decision="validate"),
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
            "geometry": {"axb"},
            "deletion": {"obs"},
            "compensation": {"complement"},
            "update_scope": {"layer"},
            "structure": {"none", "post"},
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
    return strategy
