from __future__ import annotations

import itertools
import time
from pruning.evaluation import loss_change


def targets(schedule, maximum):
    if schedule.mode == "greedy":
        return itertools.repeat(None)
    if schedule.get("targets") is not None:
        return list(schedule.targets)
    fine = schedule.get("fine_spacing", schedule.get("spacing", .05))
    until = schedule.get("fine_until", maximum)
    coarse = schedule.get("coarse_spacing", fine)
    values, value = [], fine
    while value <= min(until, maximum) + 1e-9:
        values.append(round(value, 10))
        value += fine
    value = (int((until + 1e-9) / coarse) + 1) * coarse
    while value <= maximum + 1e-9:
        values.append(round(value, 10))
        value += coarse
    return values


def run_path(problem, method, protocol, schedule, recorder):
    accepted_state = problem.snapshot()
    initial = accepted_state if schedule.mode == "independent" else None
    if getattr(method, "uses_selection", False) and protocol.candidate_selection == "loss":
        problem.selection_reference()
    denominator = problem.total
    model_total = problem.model_parameters()
    validation_reference = problem.losses("validation") if protocol.selection_check else None
    reporting_reference, dense = problem.profile()

    def model_removed(remaining):
        return 100 * (denominator - remaining + problem.outside_removed()) / model_total

    recorder.result.update(parameter_denominator=denominator, total_model_parameters=model_total,
                           selection=problem.split_metadata("validation") | {
                               "enabled": validation_reference is not None, "rho": protocol.rho,
                               "n_sigma": protocol.n_sigma},
                           heldout=problem.split_metadata("reporting") | {
                               "reference_loss": float(reporting_reference.mean())})
    row = {"step": 0, "target_sparsity": 0., "removed_percent": 0., "frac_removed": 0.,
           "model_removed_percent": model_removed(denominator), "k_eff": problem.remaining(), "accepted": True, "admissible": True,
           "heldout_loss": float(reporting_reference.mean()), "heldout_damage_percent": 0.,
           "heldout_variance": float(reporting_reference.var()),
           "test_loss": float(reporting_reference.mean()), "seconds": 0.}
    if validation_reference is not None:
        row.update(selection_loss=float(validation_reference.mean()), loss_damage_percent=0.,
                   val_loss=float(validation_reference.mean()))
    recorder.record(row)
    accepted = row
    failed = False
    reason = "targets_exhausted"
    for index, target in enumerate(targets(schedule, schedule.get("maximum", protocol.max_removed))):
        if index >= protocol.max_steps or (target is None and row["frac_removed"] >= protocol.max_removed):
            reason = "max_steps" if index >= protocol.max_steps else "max_fraction"
            break
        before = accepted_state if not failed else problem.snapshot()
        if schedule.mode == "independent":
            problem.restore(initial)
        started = time.perf_counter()
        sampling = problem.prepare(index)
        active = {"step": index + 1, "target_sparsity": target, "status": "running",
                  "starting_removed_percent": 100 * (denominator - problem.remaining()) / denominator,
                  **sampling}
        recorder.result["runs"].append(active)
        recorder.active = active
        recorder.write()
        try:
            details = method.propose(problem, target, index, recorder.progress)
            if details is None:
                problem.restore(before)
                active["status"] = "no_candidates"
                reason = "candidate_exhaustion"
                break
            pruning_seconds = time.perf_counter() - started
            measured = time.perf_counter()
            validation = (loss_change(problem.losses("validation"), validation_reference)
                          if validation_reference is not None else None)
            selection_seconds = time.perf_counter() - measured
            measured = time.perf_counter()
            reporting = (loss_change(problem.losses("reporting"), reporting_reference)
                         if protocol.reporting_every and (index + 1) % protocol.reporting_every == 0 else None)
            heldout_seconds = time.perf_counter() - measured
            passed = validation is None or validation["delta"] <= (
                protocol.n_sigma * validation["se"] + protocol.rho * float(validation_reference.mean()))
            failed = failed or not passed
            remaining = problem.remaining()
            row = {"step": index + 1, "target_sparsity": target,
                   "k_eff": remaining, "removed_percent": 100 * (denominator - remaining) / denominator,
                   "frac_removed": (denominator - remaining) / denominator,
                   "model_removed_percent": model_removed(remaining), "accepted": not failed, "admissible": passed if validation is not None else None,
                   "seconds": pruning_seconds,
                   "selection_seconds": selection_seconds, "heldout_seconds": heldout_seconds,
                   "total_seconds": time.perf_counter() - started,
                   **{key: value for key, value in details.items() if key not in {"layers"}}}
            if reporting is not None:
                row.update(heldout_loss=reporting["loss"], heldout_damage_percent=reporting["damage_percent"],
                           heldout_delta=reporting["delta"], heldout_se=reporting["se"], test_loss=reporting["loss"],
                           heldout_variance=reporting["variance"])
            if validation is not None:
                row.update(selection_loss=validation["loss"], loss_damage_percent=validation["damage_percent"],
                           selection_delta=validation["delta"], selection_se=validation["se"],
                           val_loss=validation["loss"], val_delta=validation["delta"], val_se=validation["se"])
            active.update(status="complete", **row)
            recorder.record(row)
            print(f"{recorder.result['method']} step {index + 1}: {row['removed_percent']:.3f}% removed"
                  + (f", held-out damage {reporting['damage_percent']:.3f}%" if reporting else ""), flush=True)
            if not failed:
                accepted, accepted_state = row, problem.snapshot()
            elif protocol.stop_mode == "halt" and not schedule.get("continue_after_rejection", False):
                problem.restore(before)
                active["rolled_back"] = True
                reason = "loss_limit"
                break
        except Exception as error:
            problem.restore(before)
            active.update(status="failed", error=str(error))
            recorder.result.update(status="failed", error=str(error))
            recorder.write()
            raise
    if failed:
        problem.restore(accepted_state)
    if "heldout_loss" not in accepted:
        reporting = loss_change(problem.losses("reporting"), reporting_reference)
        accepted.update(heldout_loss=reporting["loss"], heldout_damage_percent=reporting["damage_percent"],
                        heldout_delta=reporting["delta"], heldout_se=reporting["se"], test_loss=reporting["loss"])
        recorder.run.push_log({"heldout_loss": reporting["loss"],
                               "heldout_damage_percent": reporting["damage_percent"]}, step=accepted["step"])
    if protocol.save_model:
        recorder.checkpoint(problem, accepted)
        recorder.result["model_artifact"] = "model.pt"
    problem.compact()
    if validation_reference is not None:
        problem.losses("validation")
    values, endpoint = problem.profile()
    endpoint["compaction_loss_change"] = float(values.mean()) - accepted["heldout_loss"]
    recorder.result["inference"] = {"dense": dense, "endpoint": endpoint}
    return recorder.finish(reason, accepted)
