from __future__ import annotations

import torch
from pruning.baselines.result_artifacts import write_json


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
