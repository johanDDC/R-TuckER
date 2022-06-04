import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import tensor, FloatTensor
import numpy as np
from tqdm import tqdm

from load import Data, KG_dataset
from model import R_TuckER
from utils import R_TuckEROptimizer, filter_predictions, compute_metrics, RSVRG

from tucker_riemopt import set_backend

BATCH_SIZE = (64, 64) # train_size, test_size
EMBEDDINGS_DIM = (200, 200) # entity_dim, relation_dim
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
EPOCHES = 500
LR = 1e-3# start learning rate
MANIFOLD_RANK = 200

def train_loop(model, train_dataloader, test_dataloader):
    model.train()
    for epoch in range(1, EPOCHES + 1):
        model.train()
        losses = []
        with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
            for features, targets in train_dataloader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE).float()
                predictions = model(features)
                with torch.no_grad():
                    loss = model.loss(predictions, targets)
                    losses.append(loss.item())
                model.optimize(features, targets)

                prbar.set_description(
                    f"Last loss: {np.round(losses[-1], 7)},\t"
                    f"mean loss: {np.round(np.mean(losses), 7)}"
                )
                prbar.update(1)

        losses = []
        metrics = None
        model.eval()
        with tqdm(total=len(test_dataloader), file=sys.stdout) as prbar:
            with torch.no_grad():
                for features, targets in test_dataloader:
                    features = features.to(DEVICE)
                    targets = targets.to(DEVICE).float()
                    predictions = model(features)
                    predictions, targets = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
                    losses.append(model.loss(predictions, targets).item())
                    metrics = compute_metrics(predictions.detach().cpu(), targets.detach().cpu(),
                                              [1, 3, 10], accum=metrics)
                    prbar.set_description(
                        f"Mean loss:\t {np.round(np.mean(losses), 4)} "
                        f"MRR:\t {np.round(metrics['mrr'] / len(test_dataloader.dataset), 4)}"
                    )
                    prbar.update(1)


if __name__ == '__main__':
    data = Data()
    entity_vocab = {data.entities[i]: i for i in range(len(data.entities))}
    relation_vocab = {data.relations[i]: i for i in range(len(data.relations))}

    train_dataset = KG_dataset(data.train_data, entity_vocab, relation_vocab, label_smoothing=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True)

    test_dataset = KG_dataset(data.test_data, entity_vocab, relation_vocab, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE[1], shuffle=False)

    set_backend("pytorch")
    model = R_TuckER(data, [50, 50, 50], (len(entity_vocab), len(relation_vocab), len(entity_vocab)), "cpu")
    model.init()
    model.to(DEVICE)

    train_loop(model, train_dataloader, test_dataloader)

