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
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport

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
    # inds = np.hstack([np.ones((predictions.shape[0] * predictions.shape[2], 2)) *
    #                   np.tile(np.arange(predictions.shape[0]), (predictions.shape[2], 1)).T.reshape(-1, 1),
    #                   np.tile(np.arange(predictions.shape[2]).reshape(-1, 1), (predictions.shape[0], 1))])
    # preds = predictions[inds].reshape(-1, T_k.shape[0])
    for i in range(1, predictions.shape[0]):
        preds = torch.cat([preds, predictions[i, i, :].reshape(1, -1)], dim=0)

    return preds


def train_one_batch(X, y, T_k: Tucker):
    loss_fn = nn.BCELoss()
    preds = eval_batch(X, T_k)
    return loss_fn(preds, y)


# stochastic grad setting
# def train_one_epoch(train_dataloader, T_k):
#     loss_fn = nn.BCELoss()
#     rank = T_k.rank
#     losses = []
#     grad = None
#     total_preds = torch.Tensor([], dtype=torch.float32, device=DEVICE)
#     total_targets = torch.Tensor([], dtype=torch.float32, device=DEVICE)
#     with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
#         for batch_id, (features, targets) in enumerate(train_dataloader):
#             features = features.to(DEVICE)
#             targets = targets.to(DEVICE).float()
#             total_targets = torch.cat([total_targets, targets], dim=1)
#             func = lambda T: train_one_batch(features, targets, T)
#
#             losses.append(func(T_k).item())
#             total_preds = torch.cat([total_preds, eval_batch(features, T_k)], dim=1)
#             if grad is None:
#                 grad = compute_gradient_projection(func, T_k)
#             else:
#                 grad += compute_gradient_projection(func, T_k)
#                 grad = grad.round(2 * rank)
#                 grad = Tucker(grad.core.detach(), [factor.detach() for factor in grad.factors])
#
#             prbar.set_description(
#               f"Last loss:\t {np.round(losses[-1], 7)}, "
#               f"mean loss:\t {np.round(np.mean(losses), 7)}"
#             )
#             prbar.update(1)
#
#         with torch.no_grad():
#             func = loss_fn(total_preds, total_targets)
#             grad = 1 / (grad.norm(qr_based=True)) * grad
#             alpha = __custom_line_search(func, T_k, -grad, rank, 10000)
#             # alpha = __armijo(func, T_k, -grad, rank, 10000)
#             T_k -= alpha * grad
#             T_k = T_k.round(rank)
#
#     return T_k, np.mean(losses)


# CG setting
# def train_one_epoch(train_dataloader, T_k, cg_config = None):
#     loss_fn = nn.BCELoss()
#     rank = T_k.rank
#     losses = []
#     grad = None
#     total_preds = torch.Tensor([], dtype=torch.float32, device=DEVICE)
#     total_targets = torch.Tensor([], dtype=torch.float32, device=DEVICE)
#     if cg_config is None:
#         cg_config = {
#             "prev_grad_norm": None,
#             "curr_grad_norm": None,
#             "alpha": 10000,
#             "conj_dir": None
#         }
#     with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
#         for batch_id, (features, targets) in enumerate(train_dataloader):
#             features = features.to(DEVICE)
#             targets = targets.to(DEVICE).float()
#             total_targets = torch.cat([total_targets, targets], dim=1)
#             func = lambda T: train_one_batch(features, targets, T)
#
#             losses.append(func(T_k).item())
#             total_preds = torch.cat([total_preds, eval_batch(features, T_k)], dim=1)
#             if grad is None:
#                 grad = compute_gradient_projection(func, T_k)
#             else:
#                 grad += compute_gradient_projection(func, T_k)
#                 grad = grad.round(2 * rank)
#                 grad = Tucker(grad.core.detach(), [factor.detach() for factor in grad.factors])
#
#             prbar.set_description(
#               f"Last loss:\t {np.round(losses[-1], 7)}, "
#               f"mean loss:\t {np.round(np.mean(losses), 7)}"
#             )
#             prbar.update(1)
#
#         with torch.no_grad():
#             if cg_config["conj_dir"] is None:
#                 cg_config["conj_dir"] = -grad
#                 cg_config["curr_grad_norm"] = grad.norm(qr_based=True)
#             if cg_config["curr_grad_norm"] is None:
#                 cg_config["curr_grad_norm"] = grad.norm(qr_based=True)
#                 beta = cg_config["curr_grad_norm"] / cg_config["prev_grad_norm"]
#                 cg_config["conj_dir"] = -grad + (beta ** 2) * vector_transport(None, T_k, cg_config["conj_dir"])
#                 cg_config["conj_dir"] = cg_config["conj_dir"].round(grad.rank)
#
#             func = loss_fn(total_preds, total_targets)
#             alpha = cg_config["alpha"] = __custom_line_search(func, T_k, cg_config["conj_dir"], rank, cg_config["alpha"])
#             T_k += alpha * cg_config["conj_dir"]
#             T_k = T_k.round(rank)
#             cg_config["prev_grad_norm"] = cg_config["curr_grad_norm"]
#             cg_config["curr_grad_norm"] = None
#
#     return T_k, np.mean(losses), cg_config


# SVRG setting
def train_one_epoch(train_dataloader, T_k, svrg_config = None):
    rank = T_k.rank
    losses = []
    full_grad = None
    if svrg_config is None:
        svrg_config = {
            "alpha": 10000,
            "memory": 10
        }
    # idx = np.random.choice(np.arange(0, len(train_dataloader)), svrg_config["memory"], False)
    idx = [1, 3]
    with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
        for batch_id, (features, targets) in enumerate(train_dataloader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).float()
            func = lambda T: train_one_batch(features, targets, T)

            losses.append(func(T_k).item())
            # if full_grad is None:
            full_grad = compute_gradient_projection(func, T_k)
            T_k -= full_grad
            T_k = T_k.round(rank)
            # else:
            #     full_grad += compute_gradient_projection(func, T_k)
            #     full_grad = full_grad.round(2 * rank)
            #     full_grad = Tucker(full_grad.core.detach(), [factor.detach() for factor in full_grad.factors])

            prbar.set_description(
                f"Last loss:\t {np.round(losses[-1], 7)}, "
                f"mean loss:\t {np.round(np.mean(losses), 7)}"
            )
            prbar.update(1)
            if batch_id > 4:
                break

    alphas = []
    with tqdm(total=svrg_config["memory"], file=sys.stdout) as prbar:
        for batch_id, (features, targets) in enumerate(train_dataloader):
            if batch_id in idx:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE).float()
                func = lambda T: train_one_batch(features, targets, T)
                grad = compute_gradient_projection(func, T_k)
                v = compute_gradient_projection(func, T_k) - \
                    vector_transport(None, T_k, grad - full_grad)
                alpha = __custom_line_search(func, T_k, -grad, rank, svrg_config["alpha"])
                T_k -= alpha * v
                T_k = T_k.round(rank)
                alphas.append(alpha)
                prbar.update(1)
        svrg_config["alpha"] = np.mean(alphas)

    return T_k, np.mean(losses), svrg_config


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
    config = None
    for epoch in range(start_epoch, EPOCHES + start_epoch):
        T_k, loss, config = train_one_epoch(train_dataloader, T_k, config)
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
    T_0 = Tucker(torch.zeros([MANIFOLD_RANK] * 3, dtype=torch.float32, device=DEVICE),
                 [torch.zeros((len(entity_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE),
                  torch.zeros((len(relation_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE),
                  torch.zeros((len(entity_vocab), MANIFOLD_RANK), dtype=torch.float32, device=DEVICE)])
    # uniform_(T_0.core, -1, 1)
    # xavier_normal_(T_0.factors[0])
    # xavier_normal_(T_0.factors[1])
    # xavier_normal_(T_0.factors[2])

    T = train(train_dataloader, test_dataloader, T_0)
