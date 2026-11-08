from __future__ import annotations

import math
import torch
from pruning.baselines.result_artifacts import write_json


class Recorder:
    def __init__(self, run, metadata):
        self.run = run
        self.result = {"result_schema": 5, "status": "running", **metadata,
                       "curve": [], "runs": []}
        self.active = None

    def write(self):
        write_json(self.run.path / "result.json", self.result)

    def record(self, row):
        self.result["curve"].append(row)
        values = {key: float(value) for key, value in row.items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value)}
        self.run.push_log(values, step=len(self.result["curve"]) - 1)
        self.write()

    def progress(self, completed, history, canonical_savings, seconds):
        if self.active is not None:
            self.active.update(completed_layers=completed, layers=history,
                               canonical_savings=canonical_savings, seconds=seconds)
            self.write()

    def finish(self, reason, accepted):
        self.result["stop_index"] = self.result["curve"].index(accepted)
        self.result["rejected_candidate"] = next((row for row in self.result["curve"]
                                                  if row.get("admissible") is False), None)
        self.result.update(status="complete", termination_reason=reason,
                           accepted_endpoint=accepted, summary=accepted)
        self.write()
        return {key: float(value) for key, value in accepted.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)}

    def checkpoint(self, problem, row):
        torch.save({"state": problem.snapshot(), "target_sparsity": row.get("target_sparsity"),
                    "removed_percent": row["removed_percent"]}, self.run.path / "model.pt")
