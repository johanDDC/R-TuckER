import sys

import torch
from torch.utils.data import DataLoader
from torch import tensor, FloatTensor
from time import time

import numpy as np
from tqdm import tqdm

from load import Data, KG_dataset
from model import R_TuckER
from utils import *

from tucker_riemopt import set_backend

BATCH_SIZE = (64, 64)  # train_size, test_size
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
EPOCHES = 500
LR = 400
ADAM_LR = 1e-3  # start learning rate
MANIFOLD_RANK = np.array([100, 100, 100])
MOMENTUM_BETA = 0.6


class Experiment:
    def __init__(self, data, model, **experiment_settings):
        self.data = data,
        self.model = model
        self.device = experiment_settings.get("device", "cpu")
        self.optimizer = R_TuckEROptimizer(self.model.parameters(), self.model,
                                           rank=experiment_settings.get("rank", MANIFOLD_RANK),
                                           lr=experiment_settings.get("lr", LR),
                                           adam_lr=experiment_settings.get("regular_lr", ADAM_LR),
                                           scheduler_constructor=lambda optim: torch.optim.lr_scheduler.ExponentialLR(
                                               optimizer=optim,
                                               gamma=experiment_settings.get("adam_dr", 1)),
                                           momentum_beta=experiment_settings.get("momentum_beta", MOMENTUM_BETA))
        self.decay_rate = experiment_settings.get("dr", 1)
        self.epoches = experiment_settings.get("epoches", EPOCHES)
        batch_size_train = experiment_settings.get("batch_size_train", 128)
        batch_size_eval = experiment_settings.get("batch_size_eval", 128)
        label_smoothing = experiment_settings.get("label_smoothing", 0.1)
        train_dataset = KG_dataset(data, data.train_data, label_smoothing=label_smoothing)
        val_dataset = KG_dataset(data, data.valid_data, test_set=True)
        test_dataset = KG_dataset(data, data.test_data, test_set=True)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False)
        self.model.init()
        self.model.to(self.device)

    def __train_one_epoch(self):
        dataloader_len = len(self.train_dataloader)
        losses = torch.empty(size=(dataloader_len,), device=self.device)
        norms = torch.empty(size=(dataloader_len,), device=self.device)
        with tqdm(total=dataloader_len) as prbar:
            for idx, (features, targets) in enumerate(self.train_dataloader):
                features, targets = features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                predictions, loss_fn = model(features[:, 0], features[:, 1])
                nrm = self.optimizer.fit(loss_fn, targets)
                loss = self.optimizer.loss(predictions, targets)
                norms[idx] = nrm.detach()
                losses[idx] = loss.detach()
                self.optimizer.step()
                prbar.update(1)

        return torch.mean(losses).item(), torch.mean(norms).item()

    def evaluate(self, dataloader):
        hits = [[] for _ in range(10)]
        ranks = []
        dataloader_len = len(dataloader)
        losses = torch.empty(size=(dataloader_len,), device=self.device)
        with tqdm(total=dataloader_len) as prbar:
            with torch.no_grad():
                for idx, (features, targets) in enumerate(dataloader):
                    features, targets = features.to(self.device), targets.to(self.device)
                    predictions = self.model(features[:, 0], features[:, 1])
                    loss = self.model.loss(predictions, targets)
                    losses[idx] = loss
                    fpreds, _ = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
                    _, sort_idxs = torch.sort(fpreds, dim=1, descending=True)
                    sort_idxs = sort_idxs.cpu().numpy()
                    for j in range(features.shape[0]):
                        rank = np.where(sort_idxs[j] == features[j, 2].item())[0][0]
                        ranks.append(rank + 1)
                        for hits_level in range(10):
                            hits[hits_level].append(1 if rank <= hits_level else 0)
                    prbar.update(1)

        metrics = {
            "mrr": np.mean(1. / np.array(ranks)),
            "hits_1": np.mean(hits[0]),
            "hits_3": np.mean(hits[2]),
            "hits_10": np.mean(hits[9]),
        }
        return metrics, torch.mean(losses).item()

    def train(self, baselines=None, losses=None, metrics=None, start_epoch=1, draw=False):
        self.losses = {
            "train": [],
            "norm": [],
            "val": [],
            "test": []
        } if losses is None else losses
        self.metrics = {
            "mrr": [],
            "hits_1": [],
            "hits_3": [],
            "hits_10": []
        } if metrics is None else metrics
        for epoch in range(start_epoch, self.epoches + start_epoch):
            loss, mean_norm = self.__train_one_epoch()
            torch.save(self.model, "snapshot.pt")

            _, mean_loss = self.evaluate(self.val_dataloader)
            self.losses["val"].append(mean_loss)

            if self.optimizer.lr > 100 and len(losses["val"]) > 1 and losses["val"][-1] > losses["val"][-2]:
                self.optimizer.lr *= self.decay_rate
            if self.optimizer.lr <= 100:
                self.optimizer.lr *= 0.95
            local_metrics, mean_loss = self.evaluate(self.test_dataloader)
            update_metrics(metrics, local_metrics)
            losses["train"].append(loss)
            losses["norm"].append(mean_norm)
            losses["test"].append(mean_loss)

            save_model(self.model, f"./models/rk_{self.model.rank[0]}_epoch_{epoch}",
                       losses, metrics, epoch)
            if draw:
                draw_plots(losses, metrics, baselines)
        return self


if __name__ == '__main__':
    data = Data(reverse=True)
    set_random_seed(20)
    set_backend("pytorch")
    model = R_TuckER((len(data.entities), len(data.relations)),
                     MANIFOLD_RANK)
    # model = torch.load('mdl.pt').to(DEVICE)
    #
    experiment = Experiment(data, model, dr=0.95)
    # experiment.evaluate(experiment.test_dataloader)
    experiment.train()
