from __future__ import annotations

import math
import torch
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset


class Fisher:
    """Stream square-root Fisher rows; never construct the full parameter-square matrix."""
    def __init__(self, network, inputs, *, batch_size=16, likelihood="auto", variance=1.):
        self.network, self.inputs, self.batch_size = network, inputs, batch_size
        with torch.no_grad():
            output = network.logits(network.parameters, inputs[:1])
        self.likelihood = ("bernoulli" if output.numel() == 1 else "categorical") if likelihood == "auto" else likelihood
        if self.likelihood not in {"bernoulli", "categorical", "gaussian"} or variance <= 0:
            raise ValueError("Fisher likelihood must be auto, bernoulli, categorical, or gaussian with positive variance")
        self.variance = variance

    def batches(self):
        network = self.network
        parameters = {name: value.detach() for name, value in network.parameters.items()}
        def single(values, x):
            return network.logits(values, x.unsqueeze(0)).reshape(-1)
        for (x,) in DataLoader(TensorDataset(self.inputs), batch_size=self.batch_size):
            with torch.enable_grad():
                jacobian = vmap(jacrev(single), in_dims=(None, 0))(parameters, x)
            with torch.no_grad():
                logits = network.logits(parameters, x)
                rows = torch.cat([jacobian[name].reshape(len(x), -1, value.numel())
                                  for name, value in parameters.items()], -1).double()
                if self.likelihood == "categorical":
                    probabilities = logits.double().softmax(-1).reshape(len(x), -1, logits.shape[-1])
                    rows = rows.reshape(*probabilities.shape, -1)
                    rows -= (probabilities[..., None] * rows).sum(-2, keepdim=True)
                    rows *= probabilities.sqrt()[..., None]
                    normalization = len(self.inputs) * probabilities.shape[1]
                else:
                    normalization = len(self.inputs) * rows.shape[1]
                    if self.likelihood == "bernoulli":
                        probability = logits.double().sigmoid().reshape(len(x), -1)
                        rows *= (probability * (1 - probability)).sqrt()[..., None]
                    else:
                        rows /= math.sqrt(self.variance)
                yield rows.reshape(-1, rows.shape[-1]) / math.sqrt(normalization)

    def diagonal(self):
        result = torch.zeros_like(self.network.theta)
        for rows in self.batches():
            result += rows.square().sum(0)
        return result

    def correlations(self, deleted):
        result = torch.zeros_like(self.network.theta)
        for rows in self.batches():
            result += rows.T @ (rows[:, deleted] @ self.network.theta[deleted])
        return result

    def quadratic(self, blocks, updates):
        size = len(blocks) + len(updates)
        result = self.network.theta.new_zeros((size, size))
        for rows in self.batches():
            effects = torch.stack([-(rows[:, block] @ self.network.theta[block]) for block in blocks], 1)
            design = torch.cat((effects, rows[:, updates]), 1)
            result += design.T @ design
        values, vectors = torch.linalg.eigh((result + result.T) / 2)
        return (vectors * values.clamp_min(0)) @ vectors.T

    def matrix(self, coordinates):
        result = self.network.theta.new_zeros((len(coordinates), len(coordinates)))
        for rows in self.batches():
            local = rows[:, coordinates]
            result += local.T @ local
        return (result + result.T) / 2
