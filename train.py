import argparse
import wandb
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.regularization import SimpleDecreasingPolicy, RegularizationCoeffPolicy, CyclicDecreasingPolicy
from tucker_riemopt import set_backend

from src.data.Dataset import KG_dataset
from src.utils.storage import Losses, Metrics, StateDict
from src.data.Data import Data
from src.utils.utils import set_random_seed, filter_predictions, get_rank_approximation, Timer
from src.utils.metrics import metrics
from configs.base_config import Config


def define_optimizer(model, cfg):
    if MODE == "symmetric":
        param_list = nn.ParameterList([model.core, model.E.weight, model.R.weight])
    else:
        param_list = nn.ParameterList([model.core, model.S.weight, model.R.weight, model.O.weight])
    if OPT == "rsgd":
        opt = RSGDwithMomentum(param_list, cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate,
                               cfg.train_cfg.momentum_beta)
    elif OPT == "rgd":
        opt = RGD(param_list, cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate)
    elif OPT == "adam":
        opt = SFTuckerAdam(param_list, cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate, step_velocity=1)
        SFTuckerAdam
    else:
        raise NotImplementedError("Such optimization method is not implemented")
    return opt


def extract_tensor(model: nn.Module):
    if MODE == "symmetric":
        x_k = SFTucker(model.core.data, [model.R.weight], num_shared_factors=2, shared_factor=model.E.weight)
    else:
        x_k = Tucker(model.core.data, [model.R.weight, model.S.weight, model.O.weight])
    return x_k


def wandb_log(state, **kwargs):
    wandb.log({
            "train_loss": state.losses.train[-1],
            "test_loss": state.losses.test[-1],
            "val_loss": state.losses.val[-1],

            "test_mrr": state.metrics.mrr.test[-1],
            "val_mrr": state.metrics.mrr.val[-1],

            "test_hits@1": state.metrics.hits_1.test[-1],
            "val_hits@1": state.metrics.hits_1.val[-1],

            "test_hits@3": state.metrics.hits_3.test[-1],
            "val_hits@3": state.metrics.hits_3.val[-1],

            "test_hits@10": state.metrics.hits_10.test[-1],
            "val_hits@10": state.metrics.hits_10.val[-1],

            "grad_norm": state.losses.norms[-1],
            **kwargs
        })
    


def train_one_epoch(model, optimizer, criterion, train_loader, regularization_coeff=1e-4):
    model.train()
    dataloader_len = len(train_loader)
    train_loss = torch.zeros((1,), device=DEVICE)
    train_grad_norm = torch.zeros((1,), device=DEVICE)
    with tqdm(total=dataloader_len) as prbar:
        for batch_id, (features, targets) in enumerate(train_loader):
            features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            score_fn = model(features[:, 0], features[:, 1])
            loss_fn = lambda T: criterion(score_fn(T), targets)# + regularization_coeff * T.norm()
            x_k = extract_tensor(model)

            grad_norm = optimizer.fit(loss_fn, x_k)
            optimizer.step()
            train_grad_norm += grad_norm.detach()
            train_loss += optimizer.loss.detach()

            optimizer.zero_grad(set_to_none=True)
            prbar.set_description(f"{train_loss / (batch_id + 1)},\t{train_grad_norm / (batch_id + 1)}")
            prbar.update(1)

    return train_loss.item() / dataloader_len, train_grad_norm.item() / dataloader_len


@torch.no_grad()
def evaluate(model, criterion, dataloader):
    model.eval()
    dataloader_len = len(dataloader)
    val_loss = torch.zeros((1,), device=DEVICE)
    val_metrics = {
        "mrr": 0,
        "hits@1": 0,
        "hits@3": 0,
        "hits@10": 0,
    }
    denom = 0
    with tqdm(total=dataloader_len) as prbar:
        for batch_id, (features, targets) in enumerate(dataloader):
            features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            score_fn = model(features[:, 0], features[:, 1])
            x_k = extract_tensor(model)
            
            predictions = score_fn(x_k)
            loss = criterion(predictions, targets)
            val_loss += loss.detach()
            filtered_preds, _ = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
            
            batch_metrics = metrics(filtered_preds, targets)
            for key in batch_metrics.keys():
                val_metrics[key] += batch_metrics[key]
            denom += features.shape[0]
            prbar.update(1)

    for key in val_metrics.keys():
        val_metrics[key] = val_metrics[key].item() / denom
    return val_metrics, val_loss / dataloader_len


def train(model, optimizer, train_loader, val_loader, test_loader, config: Config, regulizer: RegularizationCoeffPolicy,
          scheduler=None, ) -> StateDict:
    timer = Timer()
    losses = Losses() if not config.state_dict else config.state_dict.losses
    metrics = Metrics() if not config.state_dict else config.state_dict.metrics
    num_epoches = config.train_cfg.num_epoches
    start_epoch = 1 if not config.state_dict else config.state_dict.last_epoch

    criterion = nn.BCELoss(reduction="mean")
    prev_val_mrr = evaluate(model, criterion, val_loader)[0]["mrr"]
    for epoch in range(start_epoch, num_epoches + start_epoch):
        regularization_coeff = regulizer.step()
        with timer:
            train_loss, train_norm = train_one_epoch(model, optimizer, criterion, train_loader,
                                                    regularization_coeff=regularization_coeff)
        epoch_time = timer.time

        val_metrics, val_loss = evaluate(model, criterion, val_loader)
        with timer:
            test_metrics, test_loss = evaluate(model, criterion, test_loader)
        eval_time = timer.time

        metrics.update(val_metrics, "val")
        metrics.update(test_metrics, "test")
        losses.update(train_loss, train_norm, val_loss, test_loss)

        state = StateDict(model.state_dict(), losses, metrics, epoch, optimizer.state_dict(), scheduler.state_dict())
        state.save(config.train_cfg.checkpoint_path, "snapshot", add_epoch=False)
        cur_val_mrr = val_metrics["mrr"]
        if cur_val_mrr - prev_val_mrr > 5e-4:
            prev_val_mrr = cur_val_mrr
            state.save(config.train_cfg.checkpoint_path, f"rk_{model.rank[1]}")

        if scheduler is not None:
            scheduler.step()

        wandb_log(state, lr=optimizer.param_groups[0]["lr"], reg_coeff=regularization_coeff,
                  epoch_time=epoch_time, eval_time=eval_time)

    return state


def tune(model, optimizer, train_loader, val_loader, test_loader, config: Config, regulizer: RegularizationCoeffPolicy,
         scheduler=None, ) -> StateDict:
    losses = Losses() if not config.state_dict else config.state_dict.losses
    metrics = Metrics() if not config.state_dict else config.state_dict.metrics
    tune_cfg = config.tune_cfg
    num_runs = tune_cfg.num_tunning_runs
    num_epoches = tune_cfg.num_run_epochs
    start_epoch = 1

    criterion = nn.BCELoss(reduction="mean")
    prev_val_mrr = evaluate(model, criterion, val_loader)[0]["mrr"]
    rank = list(config.model_cfg.manifold_rank)
    for run in range(1, num_runs + 1):
        rank[0] += tune_cfg.relation_rank_inc
        rank[1] += tune_cfg.entity_rank_inc
        rank[2] += tune_cfg.entity_rank_inc
        model = get_rank_approximation(model, rank)
        optimizer = define_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.train_cfg.scheduler_step) \
            if scheduler is not None else None
        optimizer.rank = rank
        optimizer.momentum = optimizer.direction = None
        model.to(DEVICE)
        for epoch in range(start_epoch, num_epoches + start_epoch):
            regularization_coeff = regulizer.step()
            train_loss, train_norm = train_one_epoch(model, optimizer, criterion, train_loader,
                                                     regularization_coeff=regularization_coeff)

            val_metrics, val_loss = evaluate(model, criterion, val_loader)
            test_metrics, test_loss = evaluate(model, criterion, test_loader)

            metrics.update(val_metrics, "val")
            metrics.update(test_metrics, "test")
            losses.update(train_loss, train_norm, val_loss, test_loss)

            state = StateDict(model.state_dict(), losses, metrics, epoch)
            state.save(config.train_cfg.checkpoint_path, "snapshot", add_epoch=False)
            cur_val_mrr = val_metrics["mrr"]
            if cur_val_mrr - prev_val_mrr > 5e-4:
                prev_val_mrr = cur_val_mrr
                state.save(config.train_cfg.checkpoint_path, f"tune_run_{run}")

            if scheduler is not None:
                scheduler.step()

            wandb_log(state, lr=optimizer.param_groups[0]["lr"], reg_coeff=regularization_coeff,
                      relation_rank=rank[0], entity_rank=rank[1])

    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="Model type", required=True)
    parser.add_argument('--seed', type=int, help="Random seed", default=20)
    parser.add_argument('--nw', type=int, help="Num workers", default=6)
    parser.add_argument('--device', type=str, help="Device", default="cuda")
    parser.add_argument('--optim', type=str, help="Optimizer", default="rsgd")
    parser.add_argument('--data', type=str, help="Dataset path", default="data/FB15k-237/")
    tune_parser = parser.add_mutually_exclusive_group(required=False)
    tune_parser.add_argument('--tune', dest="tune", action="store_true", help="Use rank tunning")
    tune_parser.add_argument('--notune', dest="tune", action='store_false', help="Do not use rank tunning")
    parser.set_defaults(tune=False)
    args = dict(vars(parser.parse_args()))
    MODE, SEED, NUM_WORKERS, DEVICE, OPT, DATA, TUNE = \
        args["mode"], args["seed"], args["nw"], args["device"], args["optim"], args["data"], args["tune"]

    if MODE == "symmetric":
        from src.model.symmetric.optim import RSGDwithMomentum, RGD, SFTuckerAdam
        from tucker_riemopt import SFTucker
        from src.model.symmetric.R_TuckER import R_TuckER
    else:
        from src.model.asymmetric.optim import RSGDwithMomentum
        from tucker_riemopt import Tucker
        from src.model.asymmetric.R_TuckER import R_TuckER

    data = Data(DATA, reverse=True)
    set_random_seed(SEED)
    set_backend("pytorch")
    cfg = Config(None)

    train_batch_size = cfg.train_cfg.train_batch_size
    test_batch_size = cfg.train_cfg.eval_batch_size

    model = R_TuckER((len(data.entities), len(data.relations)), cfg.model_cfg.manifold_rank,
                     device=DEVICE)
    model_state_dict = None
    if cfg.model_cfg.use_pretrained:
        cfg.state_dict = StateDict.load(cfg.model_cfg.pretrained_path)
        model_state_dict = cfg.state_dict.model
    model.init(model_state_dict)
    model.to(DEVICE)

    opt = define_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, cfg.train_cfg.scheduler_step)
    if TUNE:
        regulizer = CyclicDecreasingPolicy(cfg.train_cfg.base_regularization_coeff,
                                           cfg.train_cfg.num_regularizer_decreasing_steps,
                                           cfg.train_cfg.final_regularization_coeff,
                                           cfg.train_cfg.coeff_adjusting_policy)
    else:
        regulizer = SimpleDecreasingPolicy(cfg.train_cfg.base_regularization_coeff,
                                           cfg.train_cfg.num_regularizer_decreasing_steps,
                                           cfg.train_cfg.final_regularization_coeff,
                                           cfg.train_cfg.coeff_adjusting_policy)

    train_dataset = KG_dataset(data, data.train_data, label_smoothing=cfg.train_cfg.label_smoothig)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=NUM_WORKERS)

    val_dataset = KG_dataset(data, data.valid_data, test_set=True)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                                num_workers=NUM_WORKERS)

    test_dataset = KG_dataset(data, data.test_data, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=NUM_WORKERS)

    with wandb.init(project=cfg.log_cfg.project_name, entity=cfg.log_cfg.entity_name, config={
        "model": cfg.model_cfg.to_dict(),
        "train_params": cfg.train_cfg.to_dict(),
        "tune": cfg.tune_cfg.to_dict()
    }, name=cfg.log_cfg.run_name, dir=cfg.log_cfg.log_dir):
        wandb.watch(model, log=cfg.log_cfg.watch_log, log_freq=cfg.log_cfg.watch_log_freq)
        if not TUNE:
            final_state = train(model, opt, train_dataloader, val_dataloader, test_dataloader, cfg,
                                regulizer=regulizer, scheduler=scheduler)
        else:
            final_state = tune(model, opt, train_dataloader, val_dataloader, test_dataloader, cfg,
                                regulizer=regulizer, scheduler=scheduler)

    print("Final loss value:", final_state.losses.test[-1], sep="\t")
    print("Final mrr value:", final_state.metrics.mrr.test[-1], sep="\t")
    print("Final hits@1 value:", final_state.metrics.hits_1.test[-1], sep="\t")
    print("Final hits@3 value:", final_state.metrics.hits_3.test[-1], sep="\t")
    print("Final hits@10 value:", final_state.metrics.hits_10.test[-1], sep="\t")
    final_state.save(cfg.train_cfg.checkpoint_path, f"rk_{model.rank[1]}_final", add_epoch=False)
