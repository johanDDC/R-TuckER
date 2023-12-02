import os
import torch

from dataclasses import dataclass, field
from typing import List, Union


@dataclass()
class Losses:
    train: List[float] = field(default_factory=list)
    test: List[float] = field(default_factory=list)
    val: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)

    def update(self, train_loss=None, train_norm=None, val_loss=None, test_loss=None):
        self.train.append(train_loss)
        self.test.append(test_loss)
        self.val.append(val_loss)
        self.norms.append(train_norm)

    def merge(self, other_losses: "Losses"):
        self.train.extend(other_losses.train)
        self.test.extend(other_losses.test)
        self.val.extend(other_losses.val)
        self.norms.extend(other_losses.norms)


@dataclass()
class Metric:
    test: List[float] = field(default_factory=list)
    val: List[float] = field(default_factory=list)

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass()
class Metrics:
    mrr: Metric = field(default_factory=Metric)
    hits_1: Metric = field(default_factory=Metric)
    hits_3: Metric = field(default_factory=Metric)
    hits_10: Metric = field(default_factory=Metric)

    def update(self, metrics_dict: dict, type: str):
        self.mrr[type].append(metrics_dict["mrr"])
        self.hits_1[type].append(metrics_dict["hits@1"])
        self.hits_3[type].append(metrics_dict["hits@3"])
        self.hits_10[type].append(metrics_dict["hits@10"])

    def merge(self, other_metrics: "Metrics"):
        self.mrr.val.extend(other_metrics.mrr.val)
        self.mrr.test.extend(other_metrics.mrr.test)
        self.hits_1.val.extend(other_metrics.hits_1.val)
        self.hits_1.test.extend(other_metrics.hits_1.test)
        self.hits_3.val.extend(other_metrics.hits_3.val)
        self.hits_3.test.extend(other_metrics.hits_3.test)
        self.hits_10.val.extend(other_metrics.hits_10.val)
        self.hits_10.test.extend(other_metrics.hits_10.test)


@dataclass()
class StateDict:
    model: dict
    losses: Losses
    metrics: Metrics
    last_epoch: int
    optimizer: Union[dict, None]
    scheduler: Union[dict, None]

    def save(self, dir, name, add_epoch=True):
        filename = name + (f"_{self.last_epoch}" if add_epoch else "") + ".pth"
        os.makedirs(dir, exist_ok=True)
        torch.save({
            "model": self.model,
            "losses": self.losses,
            "metrics": self.metrics,
            "last_epoch": self.last_epoch,
        }, os.path.join(dir, filename))

    @classmethod
    def load(cls, name, **kwargs):
        state_dict = torch.load(f"{name}.pth", **kwargs)
        return cls(**state_dict)
