from __future__ import annotations
import gzip
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pruning.paths import ROOT
from pruning.models.base import TorchProblem
from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain
DATA = ROOT / "outputs/cache/native_baselines/berman/mnist"
CHECKPOINT = ROOT / "outputs/checkpoints/berman/simplenn_mnist_official.pt"


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x.flatten(1))


def _idx(path, images):
    with gzip.open(path, "rb") as stream:
        if images:
            _, n, h, w = struct.unpack(">IIII", stream.read(16))
            value = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8)
            return value.reshape(n, 1, h, w).float().div_(255).sub_(.1307).div_(.3081)
        _, n = struct.unpack(">II", stream.read(8))
        return torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8).long()


def mnist():
    train = (_idx(DATA / "train-images-idx3-ubyte.gz", True),
             _idx(DATA / "train-labels-idx1-ubyte.gz", False))
    test = (_idx(DATA / "t10k-images-idx3-ubyte.gz", True),
            _idx(DATA / "t10k-labels-idx1-ubyte.gz", False))
    return train, test


def loader(data, batch_size):
    return DataLoader(TensorDataset(*data), batch_size=batch_size)


def model(device):
    net = SimpleNN().to(device)
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    return net.eval().requires_grad_(False)


def linears(net):
    return net.net[0], net.net[2], net.net[4]


@torch.inference_mode()
def losses(net, batches, device):
    return torch.cat([F.cross_entropy(net(x.to(device)), y.to(device), reduction="none").cpu()
                      for x, y in batches])


@torch.inference_mode()
def accuracy(net, batches, device):
    correct = sum(int((net(x.to(device)).argmax(1) == y.to(device)).sum()) for x, y in batches)
    return correct / sum(len(y) for _, y in batches)


def nonzero(net):
    return sum(int(torch.count_nonzero(p)) for p in net.parameters())


@torch.inference_mode()
def canonicalize(net):
    canonicalize_linear_chain(linears(net), (F.relu, F.relu))


class MnistProblem(TorchProblem):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)
        super().__init__(model(self.device))
        self.train, test = mnist()
        generator = torch.Generator().manual_seed(cfg.benchmark.validation_split_seed)
        order = torch.randperm(len(test[0]), generator=generator)
        a, b = cfg.benchmark.validation_a, cfg.benchmark.validation_b
        indices = {"selection": order[:a], "validation": order[a:a+b], "reporting": order[a+b:]}
        self.splits = {name: loader((test[0][ids], test[1][ids]), cfg.benchmark.batch_size)
                       for name, ids in indices.items()}
        self.total = sum(value.numel() for value in self.model.parameters())
        self.initial_nonzero = nonzero(self.model)

    def remaining(self):
        if "pruned_coordinates" in self.extra:
            return self.masked_remaining()
        return self.total - self.initial_nonzero + nonzero(self.model)

    def losses(self, split):
        return losses(self.model, self.splits[split], self.device).double()

    def split_metadata(self, split):
        return {"dataset": "MNIST", "split": split, "count": len(self.splits[split].dataset),
                "seed": self.cfg.benchmark.validation_split_seed}
