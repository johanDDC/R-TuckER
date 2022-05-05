import torch
from torch.utils.data import DataLoader
from torch import tensor, FloatTensor
import numpy as np

from load import Data, KG_dataset
from model import R_TuckER
from utils import R_TuckEROptimizer, filter_predictions, compute_metrics, RSVRG

from tucker_riemopt import set_backend

BATCH_SIZE = (64, 64) # train_size, test_size
EMBEDDINGS_DIM = (200, 200) # entity_dim, relation_dim
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
EPOCHES = 500
LR = 10# start learning rate
MANIFOLD_RANK = 50

if __name__ == '__main__':
    data = Data()
    entity_vocab = {data.entities[i]: i for i in range(len(data.entities))}
    relation_vocab = {data.relations[i]: i for i in range(len(data.relations))}

    train_dataset = KG_dataset(data.train_data, entity_vocab, relation_vocab, label_smoothing=0.3)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True)

    test_dataset = KG_dataset(data.test_data, entity_vocab, relation_vocab, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE[1], shuffle=False)

    set_backend("pytorch")
    model = R_TuckER((len(entity_vocab), len(relation_vocab)), MANIFOLD_RANK)
    model.init(MANIFOLD_RANK)
    model.to(DEVICE)
    # torch.save(model, "model.pt")
    # model = torch.load("rk_50_epoch_5.pt")
    # model.to(DEVICE)
    # optimizer = R_TuckEROptimizer(model.parameters(), model, MANIFOLD_RANK, LR)
    optimizer = RSVRG(model.parameters(), model, MANIFOLD_RANK, LR, len(train_dataloader), memory=2)
    optimizer.idx[0] = 2
    optimizer.idx[1] = 3
    a = 4
    j = 0
    for epoch in range(1, EPOCHES + 1):
        model.train()
        losses = []
        for features, targets in train_dataloader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).float()
            optimizer.zero_grad()
            predictions, loss_fn = model(features[:, 0], features[:, 1])
            optimizer.fit(loss_fn, targets)
            loss = optimizer.loss(predictions, targets)
            losses.append(loss.item())
            print("\r", np.round(np.mean(losses), 7), sep="", end="")
            if j == a:
                optimizer.step()
                optimizer.idx[0] = 2
                optimizer.idx[1] = 3
                j = 0
                break
            j += 1
        print("\r", np.round(np.mean(losses), 7), sep="")
        continue
        optimizer.step()

        model.eval()
        total_preds = torch.Tensor().to(DEVICE)
        total_targets = torch.Tensor().to(DEVICE)
        with torch.no_grad():
            for features, targets in test_dataloader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE).float()
                predictions, _ = model(features[:, 0], features[:, 1])
                predictions, targets = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
                total_preds = torch.cat([total_preds, predictions])
                total_targets = torch.cat([total_targets, targets])
            local_metrics = compute_metrics(total_preds.detach().cpu(), total_targets.detach().cpu(),
                                            [1, 3, 10], None)
        torch.save(model, "model.pt")

