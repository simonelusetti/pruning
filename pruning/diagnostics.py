from __future__ import annotations

import torch
from pruning.baselines.result_artifacts import write_json


def compare_l0_refit(problem, method, cfg, recorder):
    import json
    import time
    from omegaconf import OmegaConf
    from pruning.evaluation import loss_change, paired
    from pruning.nnprune.factorized_fisher import complement_update
    from pruning.paths import ROOT

    settings = cfg.protocol.refit_comparison
    source = (ROOT / settings.source).resolve()
    original = json.loads((source / "result.json").read_text())
    source_cfg = OmegaConf.load(source.parent / "config.yaml")
    for current, saved in ((cfg.sampling, source_cfg.sampling), (method.options, source_cfg.branch.options)):
        if OmegaConf.to_container(current, resolve=True) != OmegaConf.to_container(saved, resolve=True):
            raise ValueError("Refit comparison must preserve the source sampling and pruning options")
    if any(cfg.benchmark[key] != source_cfg.benchmark[key] for key in ("model", "dtype", "max_layers", "seed")):
        raise ValueError("Refit comparison must preserve the source model settings")
    endpoint = original["accepted_endpoint"]
    if settings.proposal_index != endpoint["step"]:
        raise ValueError("The source checkpoint must immediately precede the compared proposal")
    started = time.perf_counter()
    recorder.result.update(diagnostic="fixed_support_refit", source_run=str(source),
                           target_sparsity=settings.target, proposal_index=settings.proposal_index,
                           quadratic_evaluation="same cached factors; before exact canonicalization",
                           solver_precision="float64 for accurate refit; float32 model evaluation",
                           comparisons={}, progress={"phase": "dense_reference"})
    recorder.write()
    dense = {split: problem.losses(split) for split in ("validation", "reporting")}
    recorder.result.update(heldout=problem.split_metadata("reporting") | {
                               "reference_loss": float(dense["reporting"].mean())},
                           selection=problem.split_metadata("validation") | {
                               "reference_loss": float(dense["validation"].mean())})
    checkpoint = torch.load(source / "model.pt", map_location=problem.device, weights_only=False, mmap=True)
    problem.restore(checkpoint["state"])
    if problem.remaining() != endpoint["k_eff"]:
        raise RuntimeError("Source checkpoint parameter accounting disagrees with its saved endpoint")
    del checkpoint
    recorder.result["starting_removed_percent"] = 100 * (problem.total - problem.remaining()) / problem.total
    sampling = problem.prepare(settings.proposal_index)
    recorder.active = {"phase": "capped_proposal", **sampling}
    recorder.result["runs"].append(recorder.active)
    recorder.result["progress"] = {"phase": "capped_proposal"}
    recorder.write()
    captured, observations = {}, {"dense": dense}

    def capture(reference, values, factors):
        captured.update(reference=reference, capped=values, factors=factors)

    def evaluate(label):
        observations[label] = {split: problem.losses(split) for split in dense}
        selection = loss_change(observations[label]["validation"], dense["validation"])
        heldout = loss_change(observations[label]["reporting"], dense["reporting"])
        result = {"removed_percent": 100 * (problem.total - problem.remaining()) / problem.total,
                  "selection": selection, "heldout": heldout,
                  "admissible": selection["delta"] <= cfg.protocol.rho * float(dense["validation"].mean())
                  + cfg.protocol.n_sigma * selection["se"]}
        recorder.result["comparisons"][label] = result
        if cfg.protocol.save_model:
            artifact = f"{label}.pt"
            torch.save({"state": problem.snapshot(), "target_sparsity": settings.target,
                        "removed_percent": result["removed_percent"]}, recorder.run.path / artifact)
            recorder.result.setdefault("models", {})[label] = artifact
        recorder.run.push_log({f"{label}_heldout_damage_percent": heldout["damage_percent"],
                               f"{label}_selection_damage_percent": selection["damage_percent"],
                               f"{label}_removed_percent": result["removed_percent"]})
        torch.save(observations, recorder.run.path / "losses.pt")
        recorder.write()
        print(f"{label}: {result['removed_percent']:.3f}% removed, held-out damage {heldout['damage_percent']:.6f}%", flush=True)

    details = method.propose(problem, settings.target, settings.proposal_index, recorder.progress, capture=capture)
    recorder.result["capped_proposal"] = details
    evaluate("capped")
    expected = next((row for row in original["curve"] if row["step"] == settings.proposal_index + 1), None)
    if expected is not None:
        recorder.result["source_reproduction"] = {
            "removal_difference_pp": recorder.result["comparisons"]["capped"]["removed_percent"] - expected["removed_percent"],
            "heldout_damage_difference_pp": recorder.result["comparisons"]["capped"]["heldout"]["damage_percent"]
            - expected["heldout_damage_percent"]}
    accurate, records = {}, []
    recorder.result["progress"] = {"phase": "accurate_refit"}
    for name, capped in captured["capped"].items():
        a, b = (factor.double() for factor in captured["factors"][name])
        a, b = (a + a.T) / 2, (b + b.T) / 2
        reference = captured["reference"][name].double()
        deleted = capped.eq(0) & method.penalized[name]
        difference = capped.double() - reference
        capped_damage = .5 * float((difference * (b @ difference @ a)).sum())
        fixed = torch.where(deleted, -reference, 0.)
        rhs_norm = (b @ fixed @ a).masked_fill(deleted, 0.).norm()
        capped_residual = (b @ difference @ a).masked_fill(deleted, 0.).norm()

        def progress(iteration, residual):
            if iteration % 100 == 0:
                print(f"Accurate refit {name}: CG {iteration}, residual {residual:.3g}", flush=True)

        delta, record = complement_update(reference, a, b, deleted, rtol=settings.rtol,
                                          max_iterations=settings.max_iterations, progress=progress)
        accurate[name] = (reference + delta).masked_fill(deleted, 0.)
        if record["predicted_fisher_damage"] > capped_damage + 1e-10 * max(1., abs(capped_damage)):
            raise RuntimeError(f"Accurate refit increased the fixed-support quadratic: {name}")
        records.append({"module": name, **record, "fixed_deletions": int(deleted.sum()),
                        "additional_zeros_outside_D": int((accurate[name].eq(0) & ~deleted & method.penalized[name]).sum()),
                        "capped_fisher_damage": capped_damage,
                        "capped_relative_residual_same_rhs": float(capped_residual / rhs_norm) if float(rhs_norm) else 0.})
        recorder.result["refit_modules"] = records
        recorder.result["progress"] = {"phase": "accurate_refit", "completed_modules": len(records),
                                       "seconds": time.perf_counter() - started}
        recorder.write()
    cleanup = method.install(problem, accurate)
    recorder.result["accurate_cleanup"] = cleanup
    evaluate("converged")
    delta, se = paired(observations["converged"]["reporting"], observations["capped"]["reporting"])
    reference_loss = float(dense["reporting"].mean())
    recorder.result.update(status="complete", progress={"phase": "complete"},
                           elapsed_seconds=time.perf_counter() - started,
                           heldout_converged_minus_capped={"delta": delta, "se": se,
                               "damage_difference_pp": 100 * delta / reference_loss,
                               "two_se_pp": 200 * se / reference_loss})
    metrics = {"capped_heldout_damage_percent": recorder.result["comparisons"]["capped"]["heldout"]["damage_percent"],
               "converged_heldout_damage_percent": recorder.result["comparisons"]["converged"]["heldout"]["damage_percent"],
               "capped_fisher_damage": sum(record["capped_fisher_damage"] for record in records),
               "converged_fisher_damage": sum(record["predicted_fisher_damage"] for record in records),
               "heldout_difference_pp": 100 * delta / reference_loss}
    recorder.result["summary"] = metrics
    recorder.write()
    return metrics


class ProposalTrace:
    """Observe neuron proposals without feeding counterfactual scores into pruning."""

    def __init__(self, problem, recorder, reference):
        self.problem, self.recorder = problem, recorder
        self.reference = float(reference.mean())
        self.rows, self.baseline, self.progress = [], {}, {}
        self.largest_jump = 0.
        self.recorder.result["diagnostics"] = {
            "trace": "diagnostics.json", "start_state": "proposal_start.pt",
            "rejected_state": "rejected_proposal.pt", "largest_jump": "largest_jump_layer.pt",
            "activation_probe_positions_per_evaluation_batch": 64,
            "loss_split": "validation", "large_jump_fraction": .01}

    def write(self):
        write_json(self.recorder.run.path / "diagnostics.json", {"rows": self.rows, "progress": self.progress})

    def fisher_progress(self, completed, _covariance, counts, seconds):
        if completed % 8 and completed != self.problem.sampling.b_sequences:
            return
        self.progress = {"phase": "fisher", "step": self.step,
                         "completed_sequences": completed, "seconds": seconds,
                         "observations_per_module": min(counts.values())}
        self.write()
        print(f"TRACE Fisher step={self.step}: {completed} sequences, {seconds:.1f}s", flush=True)

    @torch.inference_mode()
    def measure(self, stage, layer=None, surviving=None, **metadata):
        captured, handles = {}, []
        layers = self.problem.model.model.decoder.layers
        indices = range(self.problem.layers_count) if stage in {"start", "canonicalized"} else (layer,)

        def capture(index, kind, value):
            value = value.detach().reshape(-1, value.shape[-1])
            positions = torch.linspace(0, len(value) - 1, min(64, len(value)), device=value.device).long()
            sample = value[positions]
            captured.setdefault((index, kind), []).append((sample > 0).cpu() if kind == "gates" else sample.float().cpu())

        for index in indices:
            handles.extend((
                layers[index].fc1.register_forward_pre_hook(
                    lambda _m, args, index=index: capture(index, "input", args[0])),
                layers[index].fc1.register_forward_hook(
                    lambda _m, _args, value, index=index: capture(index, "gates", value)),
                layers[index].fc2.register_forward_hook(
                    lambda _m, _args, value, index=index: capture(index, "output", value))))
        try:
            losses = self.problem.losses("validation")
        finally:
            for handle in handles:
                handle.remove()
        frame = {key: torch.cat(values) for key, values in captured.items()}
        mean = float(losses.mean())
        row = {"step": self.step, "target_sparsity": self.target, "stage": stage,
               "layer": layer, "selection_loss": mean,
               "damage_percent": 100 * (mean / self.reference - 1),
               "delta_from_step_start": mean - self.start_loss,
               "delta_from_layer_start": mean - self.prefix_loss,
               **metadata}
        if layer is not None:
            current = frame[layer, "input"]
            initial = self.frozen[layer, "input"]
            row["input_drift_from_step_start"] = float((current - initial).norm() / initial.norm().clamp_min(1e-30))
            before = self.baseline[layer, "output"]
            row["relative_mlp_output_change"] = float((frame[layer, "output"] - before).norm() / before.norm().clamp_min(1e-30))
            live = surviving.cpu()
            row["surviving_gate_flip_fraction"] = (float(
                (frame[layer, "gates"][:, live] != self.baseline[layer, "gates"][:, live]).float().mean())
                if bool(live.any()) else 0.)
        self.rows.append(row)
        self.progress = {"phase": stage, "step": self.step, "layer": layer}
        if stage in {"start", "canonicalized"}:
            self.baseline = frame
        if stage == "canonicalized":
            self.prefix_loss = mean
        self.write()
        print(f"TRACE step={self.step} layer={layer} {stage}: loss={mean:.6f} "
              f"damage={row['damage_percent']:.3f}%", flush=True)
        return row

    def begin(self, step, target, before):
        self.step, self.target = step, target
        torch.save({"state": before, "step": step, "target_sparsity": target},
                   self.recorder.run.path / "proposal_start.pt")
        self.start_loss = self.prefix_loss = 0.
        row = self.measure("start")
        self.start_loss = self.prefix_loss = row["selection_loss"]
        row["delta_from_step_start"] = row["delta_from_layer_start"] = 0.
        self.frozen = dict(self.baseline)
        self.write()

    def archive_layer(self, layer, snapshot, incoming, outgoing, revised_in, revised_out,
                      cuts, record, row):
        jump = row["delta_from_layer_start"]
        if jump > max(self.largest_jump, .01 * self.reference):
            self.largest_jump = jump
            torch.save({"step": self.step, "layer": layer, "factors": snapshot,
                        "incoming": incoming, "outgoing": outgoing,
                        "revised_incoming": revised_in, "revised_outgoing": revised_out,
                        "cuts": cuts, "record": record, "trace": row},
                       self.recorder.run.path / "largest_jump_layer.pt")

    def finish(self, passed):
        if not passed:
            torch.save({"state": self.problem.snapshot(), "step": self.step,
                        "target_sparsity": self.target}, self.recorder.run.path / "rejected_proposal.pt")
