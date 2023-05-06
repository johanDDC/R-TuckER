import wandb
import torch
import torch.nn as nn

from dataclasses import asdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.Dataset import KG_dataset
# from src.model.asymmetric.optim import SGDmomentum
from src.model.symmetric.optim import SGDmomentum
from src.utils.storage import Losses, Metrics, StateDict
from tucker_riemopt import set_backend
from tucker_riemopt.symmetric.tucker import Tucker
# from tucker_riemopt import Tucker

from src.data.Data import Data
# from src.model.asymmetric.R_TuckER import R_TuckER
from src.model.symmetric.R_TuckER import R_TuckER
from src.utils.utils import set_random_seed, filter_predictions, draw_plots
from src.utils.metrics import metrics
from configs.base_config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DEVICE = "cpu"
# DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, optimizer, criterion, train_loader, regular_opt=None, batch_scheduler=None):
    model.train()
    dataloader_len = len(train_loader)
    train_loss = torch.zeros((1,), device=DEVICE)
    train_grad_norm = torch.zeros((1,), device=DEVICE)
    with tqdm(total=dataloader_len) as prbar:
        for batch_id, (features, targets) in enumerate(train_loader):
            features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            score_fn = model(features[:, 0], features[:, 1])
            loss_fn = lambda T: criterion(score_fn(T), targets)
            x_k = Tucker(model.core.data, [model.R.weight], num_symmetric_modes=2, symmetric_factor=model.E.weight)
            # x_k = Tucker(model.core.data, [model.R.weight, model.S.weight, model.O.weight])

            grad_norm = optimizer.fit(loss_fn, x_k)
            optimizer.step()
            train_grad_norm += grad_norm.detach()
            train_loss += optimizer.loss.detach()

            if regular_opt is not None:
                regular_opt.step()
                regular_opt.zero_grad(set_to_none=True)

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
            x_k = Tucker(model.core.data, [model.R.weight], num_symmetric_modes=2, symmetric_factor=model.E.weight)
            # x_k = Tucker(model.core.data, [model.R.weight, model.S.weight, model.O.weight])
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


def train(model, optimizer, train_loader, val_loader, test_loader, config: Config,
          epoch_scheduler=None, batch_scheduler=None, regular_optimizer=None, baselines=None, draw=False) -> StateDict:
    losses = Losses() if not config.state_dict else config.state_dict.losses
    metrics = Metrics() if not config.state_dict else config.state_dict.metrics
    num_epoches = config.train_cfg.num_epoches
    start_epoch = 1 if not config.state_dict else config.state_dict.last_epoch

    criterion = nn.BCELoss(reduction="mean")
    prev_val_mrr = evaluate(model, criterion, val_loader)[0]["mrr"]
    for epoch in range(start_epoch, num_epoches + start_epoch):
        train_loss, train_norm = train_one_epoch(model, optimizer, criterion, train_loader,
                                                 batch_scheduler=batch_scheduler, regular_opt=regular_optimizer)

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
            state.save(config.train_cfg.checkpoint_path, f"rk_{model.rank[1]}")

        if epoch_scheduler is not None:
            epoch_scheduler.step()

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
            "lr": optimizer.param_groups[0]["lr"]
        })

    return state


if __name__ == '__main__':
    data = Data(reverse=True)
    set_random_seed(20)
    set_backend("pytorch")
    cfg = Config(None)
    num_workers = 4

    train_batch_size = cfg.train_cfg.train_batch_size
    test_batch_size = cfg.train_cfg.eval_batch_size

    model = R_TuckER((len(data.entities), len(data.relations)), cfg.model_cfg.manifold_rank,
                     batch_size=64, device=DEVICE)
    model_state_dict = None
    if cfg.model_cfg.use_pretrained:
        cfg.state_dict = StateDict.load(cfg.model_cfg.pretrained_path)
        model_state_dict = cfg.state_dict.model
    model.init(model_state_dict)
    model.to(DEVICE)

    opt = SGDmomentum(nn.ParameterList([model.core, model.E.weight, model.R.weight]),
                      cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate,
                      cfg.train_cfg.momentum_beta, cfg.train_cfg.armijo_slope,
                      cfg.train_cfg.armijo_increase, cfg.train_cfg.armijo_decrease)
    # opt = SGDmomentum(nn.ParameterList([model.core, model.S.weight, model.R.weight, model.O.weight]),
    #                   cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate,
    #                   cfg.train_cfg.momentum_beta, cfg.train_cfg.armijo_slope,
    #                   cfg.train_cfg.armijo_increase, cfg.train_cfg.armijo_decrease)
    regular_opt = torch.optim.SGD(model.bn0.parameters(), lr=1e-4, momentum=0.9)
    epoch_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 50], 0.5)
    # epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train_cfg.num_epoches, 100)

    train_dataset = KG_dataset(data, data.train_data, label_smoothing=cfg.train_cfg.label_smoothig)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=num_workers)

    val_dataset = KG_dataset(data, data.valid_data, test_set=True)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                                num_workers=num_workers)

    test_dataset = KG_dataset(data, data.test_data, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    with wandb.init(project="R_TuckER", entity="johan_ddc_team", config={
        "model": cfg.model_cfg.to_dict(),
        "train_params": cfg.train_cfg.to_dict()
    }, name="symmetric_fix_30_qr", dir="./wandb_logs", ):
        wandb.watch(model, log="all", log_freq=500)
        final_state = train(model, opt, train_dataloader, val_dataloader, test_dataloader, cfg,
                            regular_optimizer=regular_opt, epoch_scheduler=epoch_scheduler, draw=True)

    print("Final loss value:", final_state.losses.test[-1], sep="\t")
    print("Final mrr value:", final_state.metrics.mrr.test[-1], sep="\t")
    print("Final hits@1 value:", final_state.metrics.hits_1.test[-1], sep="\t")
    print("Final hits@3 value:", final_state.metrics.hits_3.test[-1], sep="\t")
    print("Final hits@10 value:", final_state.metrics.hits_10.test[-1], sep="\t")
    final_state.save(cfg.train_cfg.checkpoint_path, f"rk_{model.rank[1]}_final", add_epoch=False)
