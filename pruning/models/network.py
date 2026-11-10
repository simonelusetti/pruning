from __future__ import annotations

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader, TensorDataset
from pruning.models.base import TorchProblem


class NetworkProblem(TorchProblem):
    """A configured nn.Module factory and train/val/test tensor-pair factory."""
    def __init__(self, cfg):
        self.cfg, self.device = cfg, torch.device(cfg.runtime.device)
        super().__init__(instantiate(cfg.benchmark.model).to(self.device).eval())
        data = instantiate(cfg.benchmark.data)
        self.train = data["train"]
        x, y = data["val"]
        generator = torch.Generator(device=x.device).manual_seed(cfg.benchmark.validation_split_seed + 1000)
        order = torch.randperm(len(x), generator=generator, device=x.device)
        first, second = order[:len(x) // 2], order[len(x) // 2:]
        self.splits = {"selection": (x[first], y[first]), "validation": (x[second], y[second]),
                       "reporting": data["test"]}
        self.total = sum(parameter.numel() for parameter in self.model.parameters())

    def remaining(self):
        return self.masked_remaining() if "pruned_coordinates" in self.extra else self.total

    @torch.inference_mode()
    def losses(self, split):
        losses = []
        mode = self.cfg.branch.options.get("likelihood", "auto")
        for x, y in DataLoader(TensorDataset(*self.splits[split]), batch_size=self.cfg.benchmark.batch_size):
            output, y = self.model(x.to(self.device)), y.to(self.device)
            output = output.logits if hasattr(output, "logits") else output
            likelihood = ("bernoulli" if output[:1].numel() == 1 else "categorical") if mode == "auto" else mode
            if likelihood == "categorical":
                loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), y.reshape(-1), reduction="none")
            elif likelihood == "bernoulli":
                loss = F.binary_cross_entropy_with_logits(output, y.to(output).reshape_as(output), reduction="none")
            else:
                variance = self.cfg.branch.options.variance
                loss = .5 * ((output - y.reshape_as(output)).square() / variance + torch.log(output.new_tensor(2 * torch.pi * variance)))
            losses.append(loss.reshape(len(x), -1).mean(1).cpu())
        return torch.cat(losses).double()

    def split_metadata(self, split):
        return {"dataset": self.cfg.benchmark.name, "split": split,
                "count": len(self.splits[split][0]), "seed": self.cfg.benchmark.validation_split_seed}
