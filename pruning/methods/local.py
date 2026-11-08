from __future__ import annotations

from dataclasses import fields
from pruning.nnprune.core import PruneConfig, refresh, candidates, evaluate_candidates, select_move
from pruning.nnprune.fisher import resolve_strategy


class LocalGreedy:
    uses_selection = True

    def __init__(self, cfg, options):
        protocol = cfg.protocol
        values = dict(options)
        weights = values.pop("weights", False)
        if cfg.branch.name == "discrete":
            penalties = {f"fisher_{key}": values.pop(key) for key in (
                "lambda_u", "lambda_d", "mu", "pool", "time_limit", "gap", "bound", "log")}
            values.update(penalties, weight_per_site=0, struct_kinds=("discrete",),
                          fisher_strategy=resolve_strategy("discrete"))
        else:
            values["struct_kinds"] = ("classic",)
            if not weights:
                values["weight_per_site"] = 0
        if cfg.benchmark.get("supervisor_protocol", False):
            values.update(screen_n=None, select_admissibility=False, tie_break="k_eff")
            if weights:
                values["weight_score"] = "fdist"
        values.update(n_sigma=protocol.n_sigma, tol_frac=protocol.rho,
                      select_split="valA", stop_split="valB", seed=cfg.benchmark.seed,
                      stop_mode=protocol.stop_mode, max_frac=protocol.max_removed,
                      max_steps=protocol.max_steps)
        unknown = set(values) - {f.name for f in fields(PruneConfig)}
        if unknown:
            raise ValueError(f"unsupported local method options: {sorted(unknown)}")
        self.cfg = PruneConfig(**values)
        self.protocol, self.ctx, self.since = protocol, None, 0

    def propose(self, problem, target, index, progress):
        adapter, state, cfg = problem.adapter, problem.state, self.cfg
        if self.ctx is None or self.since >= cfg.refresh_every:
            self.ctx, self.since = refresh(adapter, state, cfg), 0
        for expansion in (1, cfg.expand_on_fail):
            if expansion != 1:
                self.ctx, self.since = refresh(adapter, state, cfg), 0
            moves = candidates(adapter, state, self.ctx, cfg, expand=expansion)
            if self.protocol.candidate_selection == "proxy":
                chosen = None
                for move in sorted(moves, key=lambda m: m.pred_cost):
                    proposal = adapter.apply(state, move)
                    if proposal is not None and adapter.k_eff(proposal) < adapter.k_eff(state):
                        chosen = {"move": move, "state": proposal}
                        break
                n_adm = n_ties = int(chosen is not None)
            else:
                results = evaluate_candidates(adapter, state, moves, problem.selection_reference(), cfg)
                chosen, n_adm, n_ties = select_move(results, cfg)
            if chosen is not None:
                break
        if chosen is None:
            return None
        problem.state = chosen["state"]
        self.since += 1
        if self.ctx.get("fisher_moves"):
            self.ctx = None
        move = chosen["move"]
        solver = {key: move.payload.get(key) for key in (
            "n_deleted", "n_updates", "closure_savings", "search_objective",
            "solver_objective", "solver_gap", "solver_seconds", "compensation_bound_active")}
        return {"kind": move.kind, "site": move.site, "layer": move.layer,
                "move": move.describe(), "pred_cost": move.pred_cost,
                "n_candidates": len(moves), "n_admissible": n_adm, "n_ties": n_ties, **solver}
