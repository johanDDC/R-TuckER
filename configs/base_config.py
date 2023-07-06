from dataclasses import dataclass
from typing import Union

from src.utils.storage import StateDict


@dataclass
class TrainConfig:
    train_batch_size = 512
    eval_batch_size = 512

    num_epoches = 70
    momentum_beta = 0.9
    label_smoothig = 0.1
    learning_rate = 800
    scheduler_step = 0.9958

    base_regularization_coeff = 1e-4
    final_regularization_coeff = 1e-8
    coeff_adjusting_policy = "exp"

    checkpoint_path = "checkpoints/"

    def to_dict(self):
        return {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epoches": self.num_epoches,
            "learning_rate": self.learning_rate,
            "momentum_beta": self.momentum_beta,
            "label_smoothig": self.label_smoothig,
        }


@dataclass
class ModelConfig:
    manifold_rank = (50, 50, 50)
    # manifold_rank = (50, 200, 200)

    use_pretrained = False
    pretrained_path = ""

    def to_dict(self):
        return {
            "manifold_rank": self.manifold_rank,
            "use_pretrained": self.use_pretrained,
            "pretrained_path": self.pretrained_path
        }


@dataclass
class LogConfig:
    project_name = "R_TuckER"
    entity_name = "johan_ddc_team"
    run_name = "test_global_package_fix"
    log_dir = "wandb_logs"

    watch_log_freq = 500
    watch_log = "all"


@dataclass
class Config:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    log_cfg = LogConfig()

    state_dict: Union[None, StateDict]

