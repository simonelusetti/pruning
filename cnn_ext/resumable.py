"""
prune_path_resumable: Veronica's core.prune_path, line for line, with one addition:
when a wall-clock deadline is reached it saves its whole state (network, geometry
context, rows so far, counters) to a checkpoint and returns None; the next call with
the same checkpoint file continues exactly where it stopped.  Needed because each
command in the sandbox on the user's computer is limited to about 3 minutes.

The decision logic is not modified: it calls her refresh, candidates,
evaluate_candidates, select_move, paired_stats and _row.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from nnprune.core import (_row, candidates, evaluate_candidates, paired_stats, refresh,  # noqa: F401
                          select_move)


def prune_path_resumable(adapter, state0, cfg, label, ckpt, deadline):
    L0 = adapter.per_sample(state0, cfg.select_split, cfg.criterion)
    separate_stop = cfg.stop != cfg.select_split
    S0 = adapter.per_sample(state0, cfg.stop, cfg.criterion) if separate_stop else None
    k0 = adapter.k_eff(state0)
    if os.path.exists(ckpt):
        C = torch.load(ckpt, weights_only=False)
        state, ctx, since, rows = C["state"], C["ctx"], C["since"], C["rows"]
        fails_at, step0, elapsed = C["fails_at"], C["step"], C["elapsed"]
        torch.set_rng_state(C["rng"])
    else:
        torch.manual_seed(cfg.seed)
        state = state0
        rows = [_row(adapter, state, k0, step=0, method=label, kind="", site="",
                     layer=-1, dk=0, anch_mean=0.0, anch_se=0.0, stop_mean=0.0, stop_se=0.0, stop_pass=True,
                     n_cands=0, n_adm=0, n_ties=0, seconds=0.0)]
        ctx, since, fails_at, step0, elapsed = None, 0, None, 1, 0.0
    t0 = time.time() - elapsed

    def save(step):
        torch.save({"state": state, "ctx": ctx, "since": since, "rows": rows, "fails_at": fails_at,
                    "step": step, "elapsed": time.time() - t0, "rng": torch.get_rng_state()}, ckpt)

    done_here = 0                                   # at least one step per call, so every call makes progress
    for step in range(step0, cfg.max_steps + 1):
        if done_here and time.time() > deadline:
            save(step)
            return None
        if fails_at is not None and step - fails_at > cfg.post_stop_steps:
            break
        if 1.0 - adapter.k_eff(state) / k0 >= cfg.max_frac:
            break
        if ctx is None or since >= cfg.refresh_every:
            ctx, since = refresh(adapter, state, cfg), 0
        moves = candidates(adapter, state, ctx, cfg)
        res = evaluate_candidates(adapter, state, moves, L0, cfg)
        chosen, n_adm, n_ties = select_move(res, cfg)
        if chosen is None and cfg.expand_on_fail > 1 and (cfg.select_admissibility or not res):
            ctx, since = refresh(adapter, state, cfg), 0
            moves = candidates(adapter, state, ctx, cfg, expand=cfg.expand_on_fail)
            res = evaluate_candidates(adapter, state, moves, L0, cfg)
            chosen, n_adm, n_ties = select_move(res, cfg)
        if chosen is None:
            break
        if separate_stop:
            sm, sse = paired_stats(adapter.per_sample(chosen["state"], cfg.stop, cfg.criterion), S0)
            stop_pass = bool(sm <= cfg.n_sigma * sse + cfg.tol_abs)
            if not stop_pass and fails_at is None:
                fails_at = step
            if not stop_pass and cfg.stop_mode == "halt":
                break
        else:
            sm, sse, stop_pass = chosen["anch_mean"], chosen["anch_se"], True
        state = chosen["state"]
        since += 1
        done_here += 1
        mv = chosen["move"]
        rows.append(_row(adapter, state, k0, step=step, method=label, kind=mv.kind,
                         site=mv.site, layer=mv.layer, dk=chosen["dk"],
                         anch_mean=chosen["anch_mean"], anch_se=chosen["anch_se"],
                         stop_mean=sm, stop_se=sse, stop_pass=stop_pass,
                         n_cands=len(res), n_adm=n_adm, n_ties=n_ties,
                         seconds=time.time() - t0))
    df = pd.DataFrame(rows)
    bad = ~df["stop_pass"].values.astype(bool)
    first_fail = int(np.argmax(bad)) if bad.any() else len(df)
    df["within_stop"] = np.arange(len(df)) < first_fail
    if os.path.exists(ckpt):
        try:
            os.remove(ckpt)
        except PermissionError:                  # deleting may not be allowed in the sandbox: rename instead
            os.replace(ckpt, ckpt + ".finished")
    return df
