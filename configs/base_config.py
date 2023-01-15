from dataclasses import dataclass
from typing import Union

from src.utils.storage import StateDict


@dataclass
class TrainConfig:
    train_batch_size = 64
    eval_batch_size = 64

    num_epoches = 200
    learning_rate = 100
    momentum_beta = 0.9
    label_smoothig = 0.1

    armijo_slope = 1e-6
    armijo_increase = 0.5
    armijo_decrease = 0.75

    checkpoint_path = "checkpoints/"


@dataclass
class ModelConfig:
    manifold_rank = (10, 10, 10)

    use_pretrained = False
    pretrained_path = ""


@dataclass
class Config:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    state_dict: Union[None, StateDict]

