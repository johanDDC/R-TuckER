import torch
from torch import nn, sparse_coo_tensor
from torch.utils.data import DataLoader
from torch.nn.init import xavier_normal_, uniform_

import numpy as np
from copy import deepcopy
from tqdm import tqdm
import sys

from load import Data, KG_dataset
from utils import filter_predictions, compute_metrics

from tucker_riemopt import Tucker, set_backend, backend
from tucker_riemopt.riemopt import compute_gradient_projection

# DEVICE = "cuda"

BATCH_SIZE = (64, 64)
EPOCHES = 100
DEVICE = "cpu"
MANIFOLD_RANK = 50


def __armijo(func, x_k: Tucker, d_k, rank, previous_alpha=1):
    alpha = previous_alpha
    armijo_threshold = 1e-4 * alpha * d_k.norm(qr_based=True) ** 2
    fx = func(x_k)
    iters = 0
    while fx - func((x_k + alpha * d_k).round(rank)) < 0:
        alpha /= 2
        armijo_threshold /= 2
        iters += 1
        if iters > 50:
            return alpha
    return alpha


def __custom_line_search(func, x_k: Tucker, d_k, rank, previous_alpha=1):
    alpha = previous_alpha
    fx = func(x_k)
    fx_success = func((x_k + alpha * d_k).round(rank))
    iters = 0
    while fx <= fx_success:
        alpha /= 2
        fx_success = func((x_k + alpha * d_k).round(rank))
        iters += 1
        if iters > 30:
            return alpha

    fx = fx_success
    prev_alpha = alpha
    while fx_success <= fx and alpha < previous_alpha:
        prev_alpha = alpha
        alpha *= 1.1
        iters += 1
        fx_success = func((x_k + alpha * d_k).round(rank))
        if iters > 30:
            return alpha
    return prev_alpha


def eval_batch(X, T_k: Tucker):
    batch_size = X.shape[0]
    batch_arange = torch.arange(batch_size).to(DEVICE)
    subject_idx = torch.vstack([batch_arange, X[:, 0]])
    subject_idx = sparse_coo_tensor(subject_idx, torch.ones(subject_idx.shape[1]),
                                    (batch_size, T_k.factors[0].shape[0]), dtype=torch.float32,
                                    device=DEVICE)
    relation_idx = torch.vstack([batch_arange, X[:, 1]])
    relation_idx = sparse_coo_tensor(relation_idx, torch.ones(relation_idx.shape[1]),
                                     (batch_size, T_k.factors[1].shape[0]), dtype=torch.float32,
                                     device=DEVICE)
    predictions = T_k.k_mode_product(0, subject_idx).k_mode_product(1, relation_idx)
    predictions = torch.sigmoid(predictions.full())
    preds = predictions[0, 0, :].reshape(1, -1)
    for i in range(1, predictions.shape[0]):
        preds = torch.cat([preds, predictions[i, i, :].reshape(1, -1)], dim=0)

    return preds


def train_one_batch(X, y, T_k: Tucker):
    loss_fn = nn.BCELoss()
    preds = eval_batch(X, T_k)
    return loss_fn(preds, y)


# stochastic grad setting
def train_one_epoch(train_dataloader, T_k):
    loss_fn = nn.BCELoss()
    rank = T_k.rank
    losses = []
    grad = None
    total_preds = torch.Tensor([], dtype=torch.float32, device=DEVICE)
    total_targets = torch.Tensor([], dtype=torch.float32, device=DEVICE)
    with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
        for batch_id, (features, targets) in enumerate(train_dataloader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).float()
            total_targets = torch.cat([total_targets, targets], dim=1)
            func = lambda T: train_one_batch(features, targets, T)

            losses.append(func(T_k).item())
            total_preds = torch.cat([total_preds, eval_batch(features, T_k)], dim=1)
            if grad is None:
                grad = compute_gradient_projection(func, T_k)
            else:
                grad += compute_gradient_projection(func, T_k)
                grad = grad.round(2 * rank)
                grad = Tucker(grad.core.detach(), [factor.detach() for factor in grad.factors])

            prbar.set_description(
              f"Last loss:\t {np.round(losses[-1], 7)}, "
              f"mean loss:\t {np.round(np.mean(losses), 7)}"
            )
            prbar.update(1)

        with torch.no_grad():
            func = loss_fn(total_preds, total_targets)
            grad = 1 / (grad.norm(qr_based=True)) * grad
            alpha = __custom_line_search(func, T_k, -grad, rank, 10000)
            # alpha = __armijo(func, T_k, -grad, rank, 10000)
            T_k -= alpha * grad
            T_k = T_k.round(rank)

    return T_k, np.mean(losses)


def evaluate(test_dataloader, T_k):
  metrics = None
  loss_fn = nn.BCELoss()
  losses = []
  with tqdm(total=len(test_dataloader), file=sys.stdout) as prbar:
      with torch.no_grad():
          for features, targets in test_dataloader:
              features = features.to(DEVICE)
              targets = targets.to(DEVICE).float()
              predictions = eval_batch(features, T_k)
              predictions, targets = filter_predictions(predictions, targets, features[:, 2].reshape(-1, 1))
              losses.append(loss_fn(predictions, targets).item())
              metrics = compute_metrics(predictions.detach().cpu(), targets.detach().cpu(),
                                                [1, 3, 10], accum=metrics)

              prbar.set_description(
                  f"Mean loss:\t {np.round(np.mean(losses), 4)} "
                  f"MRR:\t {np.round(metrics['mrr'] / len(test_dataloader.dataset), 4)}"
              )
              prbar.update(1)
  for key in metrics.keys():
    metrics[key] /= len(test_dataloader.dataset)

  return metrics, np.mean(losses)


def train(train_dataloader, test_dataloader, T_0, baselines=None, losses=None, metrics=None, start_epoch=1):
    loss = {
        "train": [],
        "test": []
    } if not losses else losses
    metrics = {
        "mrr": [],
        "hits_1": [],
        "hits_3": [],
        "hits_10": []
    } if not metrics else metrics
    T_k = deepcopy(T_0)
    for epoch in range(start_epoch, EPOCHES + start_epoch):
        T_k = train_one_epoch(train_dataloader, T_k)
        local_metrics, mean_loss = evaluate(test_dataloader, T_k)


    return T_k


if __name__ == '__main__':
    data = Data()
    entity_vocab = {data.entities[i]: i for i in range(len(data.entities))}
    relation_vocab = {data.relations[i]: i for i in range(len(data.relations))}

    train_dataset = KG_dataset(data.train_data, entity_vocab, relation_vocab, label_smoothing=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=False)

    test_dataset = KG_dataset(data.test_data, entity_vocab, relation_vocab, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE[1], shuffle=False)

    set_backend("pytorch")
    T_0 = Tucker(backend.randn([MANIFOLD_RANK] * 3, dtype=torch.float32, device=DEVICE),
                 [backend.randn((len(entity_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE),
                  backend.randn((len(relation_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE),
                  backend.randn((len(entity_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE)])
    uniform_(T_0.core, -1, 1)
    xavier_normal_(T_0.factors[0])
    xavier_normal_(T_0.factors[1])
    xavier_normal_(T_0.factors[2])

    T = train(train_dataloader, test_dataloader, T_0)
