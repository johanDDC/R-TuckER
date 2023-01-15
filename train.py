import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.Dataset import KG_dataset
from src.utils.optim import SGDmomentum
from src.utils.storage import Losses, Metrics, StateDict
from tucker_riemopt import set_backend, Tucker
# from tucker_riemopt.riemopt import compute_gradient_projection

from src.data.Data import Data
from src.model.R_TuckER import R_TuckER
from src.utils.utils import set_random_seed, filter_predictions, draw_plots
from src.utils.metrics import metrics
from configs.base_config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    dataloader_len = len(train_loader)
    train_loss = 0
    train_grad_norm = 0
    with tqdm(total=dataloader_len) as prbar:
        for batch_id, (features, targets) in enumerate(train_loader):
            features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            predictions, score_fn = model(features[:, 0], features[:, 1])
            loss_fn = lambda T: criterion(score_fn(T), targets)
            loss = criterion(predictions, targets)

            x_k = Tucker(model.core.data, [model.R.weight, model.S.weight, model.O.weight])

            grad_norm = optimizer.fit(loss_fn, x_k)
            optimizer.step(loss_fn)
            train_grad_norm += grad_norm.item()
            train_loss += loss.item()

            optimizer.zero_grad()
            prbar.set_description(f"{train_loss / (batch_id + 1)}")
            prbar.update(1)

    return train_loss / dataloader_len, train_grad_norm / dataloader_len


@torch.no_grad()
def evaluate(model, criterion, dataloader):
    model.eval()
    dataloader_len = len(dataloader)
    val_loss = 0
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
            predictions = model(features[:, 0], features[:, 1])[0]
            loss = criterion(predictions, targets)
            val_loss += loss.item()
            filtered_preds, _ = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
            batch_metrics = metrics(filtered_preds, targets)
            for key in batch_metrics.keys():
                val_metrics[key] += batch_metrics[key]
            denom += features.shape[0]
            prbar.update(1)

    for key in val_metrics.keys():
        val_metrics[key] /= denom
    return val_metrics, val_loss / dataloader_len


def train(model, optimizer, train_loader, val_loader, test_loader, config: Config,
          baselines=None, draw=False) -> StateDict:
    losses = Losses() if not config.state_dict else config.state_dict.losses
    metrics = Metrics() if not config.state_dict else config.state_dict.metrics
    num_epoches = config.train_cfg.num_epoches
    start_epoch = 1 if not config.state_dict else config.state_dict.last_epoch

    criterion = nn.BCELoss()
    prev_val_mrr = evaluate(model, criterion, val_loader)[0]["mrr"]
    for epoch in range(start_epoch, num_epoches + start_epoch):
        train_loss, train_norm = train_one_epoch(model, optimizer, criterion, train_loader)

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

        if draw:
            draw_plots(state, baselines)

    return state


if __name__ == '__main__':
    data = Data(reverse=True)
    set_random_seed(20)
    set_backend("pytorch")
    cfg = Config(None)

    train_batch_size = cfg.train_cfg.train_batch_size
    test_batch_size = cfg.train_cfg.eval_batch_size

    model = R_TuckER((len(data.entities), len(data.relations)), cfg.model_cfg.manifold_rank)
    model_state_dict = None
    if cfg.model_cfg.use_pretrained:
        cfg.state_dict = StateDict.load(cfg.model_cfg.pretrained_path)
        model_state_dict = cfg.state_dict.model
    model.init(model_state_dict)
    model.to(DEVICE)

    opt = SGDmomentum(nn.ParameterList([model.core, model.S.weight, model.R.weight, model.O.weight]),
                      cfg.model_cfg.manifold_rank, cfg.train_cfg.learning_rate,
                      cfg.train_cfg.momentum_beta, cfg.train_cfg.armijo_slope,
                      cfg.train_cfg.armijo_increase, cfg.train_cfg.armijo_decrease)

    train_dataset = KG_dataset(data, data.train_data, label_smoothing=cfg.train_cfg.label_smoothig)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)

    val_dataset = KG_dataset(data, data.valid_data, test_set=True)
    val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)

    test_dataset = KG_dataset(data, data.test_data, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True)

    final_state = train(model, opt, train_dataloader, val_dataloader, test_dataloader, cfg, draw=True)
    print("Final loss value:", final_state.losses.test[-1], sep="\t")
    print("Final mrr value:", final_state.metrics.mrr.test[-1], sep="\t")
    print("Final hits@1 value:", final_state.metrics.hits_1.test[-1], sep="\t")
    print("Final hits@3 value:", final_state.metrics.hits_3.test[-1], sep="\t")
    print("Final hits@10 value:", final_state.metrics.hits_10.test[-1], sep="\t")
    final_state.save(cfg.train_cfg.checkpoint_path, f"rk_{model.rank[1]}_final", add_epoch=False)
