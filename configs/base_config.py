from dataclasses import dataclass
from typing import Union

from src.utils.storage import StateDict


@dataclass
class TrainConfig:
    train_batch_size = 512
    eval_batch_size = 512

    num_epoches = 500
    momentum_beta = 0.8
    label_smoothig = 0.1
    learning_rate = 2000
    # scheduler_step = 0.999
    scheduler_step = 0.995

    base_regularization_coeff = 1e-11
    final_regularization_coeff = 1e-16
    coeff_adjusting_policy = "linear"
    # base_regularization_coeff = 1e-8
    # final_regularization_coeff = 1e-11
    # coeff_adjusting_policy = "exp"
    num_regularizer_decreasing_steps = 300

    checkpoint_path = "checkpoints/"

    def to_dict(self):
        return {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_epoches": self.num_epoches,
            "learning_rate": self.learning_rate,
            "momentum_beta": self.momentum_beta,
            "label_smoothig": self.label_smoothig,
            "scheduler_step": self.scheduler_step,
            "base_regularization_coeff": self.base_regularization_coeff,
            "final_regularization_coeff": self.final_regularization_coeff,
            "coeff_adjusting_policy": self.coeff_adjusting_policy,
            "num_regularizer_decreasing_steps": self.num_regularizer_decreasing_steps
        }


@dataclass
class TuneConfig:
    num_tunning_runs = 5
    num_run_epochs = 30

    relation_rank_inc = 0
    entity_rank_inc = 1

    def to_dict(self):
        return {
            "num_tunning_runs": self.num_tunning_runs,
            "num_run_epochs": self.num_run_epochs,
            "relation_rank_inc": self.relation_rank_inc,
            "entity_rank_inc": self.entity_rank_inc,
        }

@dataclass
class ModelConfig:
    manifold_rank = (200, 100, 100)

    use_pretrained = False
    pretrained_path = "./checkpoints/rk_20_903"

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
    run_name = "(100, 200) FB15k-237 RTuckER square reg"
    log_dir = "wandb_logs"

    watch_log_freq = 500
    watch_log = "all"


@dataclass
class Config:
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    log_cfg = LogConfig()
    tune_cfg = TuneConfig()

    state_dict: Union[None, StateDict]

