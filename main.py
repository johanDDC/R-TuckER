import torch
from torch import nn, sparse_coo_tensor
from torch.nn.init import xavier_normal_, uniform_

import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap, random
from jax.nn.initializers import glorot_normal, glorot_uniform
from jax.config import config; config.update("jax_enable_x64", False)

import numpy as np
from scipy.sparse import coo_matrix
from copy import deepcopy

from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from load import Data, KG_dataset, numpy_collate
from utils import filter_predictions, compute_metrics, BCELoss, SparseMatrix

from tucker_riemopt import Tucker
from tucker_riemopt.riemopt import compute_gradient_projection

# DEVICE = "cuda"

BATCH_SIZE = (64, 64)
EPOCHES = 100
DEVICE = "cpu"
MANIFOLD_RANK = 50


def __armijo(func, previous_alpha=1, threshold=1e-4):
    alpha = previous_alpha
    fx = func(0)
    iters = 0
    while fx - func(alpha) < threshold:
        alpha /= 5
        threshold /= 5
        iters += 1
        if iters > 50:
            return alpha
    return alpha


def eval_batch(X, T_k: Tucker):
    batch_size = X.shape[0]
    batch_arange = jnp.arange(batch_size)
    subject_idx = jnp.vstack([batch_arange, X[:, 0]])
    subject_idx = SparseMatrix((batch_arange, X[:, 0]), jnp.ones(subject_idx.shape[1], dtype=jnp.float32),
                               (batch_size, T_k.factors[0].shape[0]))
    relation_idx = jnp.vstack([batch_arange, X[:, 1]])
    relation_idx = SparseMatrix((batch_arange, X[:, 1]), jnp.ones(relation_idx.shape[1], dtype=jnp.float32),
                               (batch_size, T_k.factors[1].shape[0]))

    # subject_idx = sparse_coo_tensor(subject_idx, torch.ones(subject_idx.shape[1]),
    #                                 (batch_size, T_k.factors[0].shape[0]), dtype=torch.float32,
    #                                 device=DEVICE)
    # relation_idx = torch.vstack([batch_arange, X[:, 1]])
    # relation_idx = sparse_coo_tensor(relation_idx, torch.ones(relation_idx.shape[1]),
    #                                  (batch_size, T_k.factors[1].shape[0]), dtype=torch.float32,
    #                                  device=DEVICE)
    predictions = T_k.k_mode_product(0, subject_idx).k_mode_product(1, relation_idx)
    predictions = jnn.sigmoid(predictions.full())
    preds = predictions[0, 0, :].reshape(1, -1)
    for i in range(1, predictions.shape[0]):
        preds = jnp.concatenate([preds, predictions[i, i, :].reshape(1, -1)], axis=0)

    return preds


def train_one_batch(X, y, T_k, dropout_prob=0.2):
    #     dropout = nn.Dropout(p=dropout_prob)
    #     T_new = Tucker(dropout(T_k.core), T_k.factors)
    preds = eval_batch(X, T_k)
    return BCELoss(preds, y, reduction="sum")


batch_losses = None


def train_one_epoch(train_dataloader, T_k, train_config=None):
    rank = MANIFOLD_RANK
    losses = []
    with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
        for batch_id, (features, targets) in enumerate(train_dataloader):
            func = lambda T: train_one_batch(features, targets, T, train_config["dropout_prob"])

            losses.append(func(T_k))

            grad = compute_gradient_projection(func, T_k)
            # alpha_func = lambda a: func((T_k - a * grad).round(rank))
            #             print(train_config["armijo_const"] * grad.norm(qr_based=True) * train_config["alpha"])
            # alpha = __armijo(alpha_func, train_config["alpha"],
            #                  train_config["armijo_const"] * grad.norm(qr_based=True) * train_config["alpha"])
            alpha = train_config["armijo_const"]
            T_k -= alpha * grad
            T_k = T_k.round(rank)

            prbar.set_description(
                f"Last loss: {np.round(losses[-1], 7)},\t"
                f"last alpha: {alpha},\t"
                #                 f"norm: {np.round(grad_norm.item(), 2)},\t"
                f"mean loss: {np.round(np.mean(losses), 7)}"
            )
            prbar.update(1)

    global batch_losses
    batch_losses = losses
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


def train(train_dataloader, val_dataloader, test_dataloader, T_0, baselines=None,
          losses=None, metrics=None, start_epoch=1, train_config=None):
    global LAST_EPOCH, EPOCHES
    losses = {
        "train": [],
        "val": [],
        "test": []
    } if not losses else losses
    metrics = {
        "mrr": [],
        "hits_1": [],
        "hits_3": [],
        "hits_10": []
    } if not metrics else metrics
    T_k = deepcopy(T_0)
    mul = 0.1
    for epoch in range(start_epoch, EPOCHES + start_epoch):
        T_k, loss = train_one_epoch(train_dataloader, T_k, train_config)
        torch.save(T_k, "save.pt")

        _, mean_loss = evaluate(val_dataloader, T_k)
        losses["val"].append(mean_loss)

        train_config["alpha"] *= mul
        if train_config["alpha"] <= 1e-3:
            mul = 0.9
        local_metrics, mean_loss = evaluate(test_dataloader, T_k)
        # update_metrics(metrics, local_metrics)
        losses["train"].append(loss)
        losses["test"].append(mean_loss)

        start_epoch += 1
        # save_meta(T_k, losses, metrics, start_epoch)
        # draw_plots(losses, metrics, baselines)

    return T_k

if __name__ == '__main__':
    data = Data(data_dir="data/WN18RR/")
    entity_vocab = {data.entities[i]: i for i in range(len(data.entities))}
    relation_vocab = {data.relations[i]: i for i in range(len(data.relations))}

    train_dataset = KG_dataset(data.train_data, entity_vocab, relation_vocab, label_smoothing=0.15)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, collate_fn=numpy_collate)

    val_dataset = KG_dataset(data.valid_data, entity_vocab, relation_vocab, test_set=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE[1], shuffle=False, collate_fn=numpy_collate)

    test_dataset = KG_dataset(data.test_data, entity_vocab, relation_vocab, test_set=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE[1], shuffle=False, collate_fn=numpy_collate)

    losses, metrics = None, None
    # try:
    #     (losses, metrics), T_0 = load_meta(f"rk_{MANIFOLD_RANK}", LAST_EPOCH)
    # except:
    T_0 = Tucker(glorot_uniform(dtype=jnp.float32)(random.PRNGKey(322), [MANIFOLD_RANK] * 3),
                 [glorot_normal(dtype=jnp.float32)(random.PRNGKey(322), (len(entity_vocab), MANIFOLD_RANK)),
                  glorot_normal(dtype=jnp.float32)(random.PRNGKey(322), (len(relation_vocab), MANIFOLD_RANK)),
                  glorot_normal(dtype=jnp.float32)(random.PRNGKey(322), (len(entity_vocab), MANIFOLD_RANK))])

    train_config = {
        "alpha": 100,
        "memory": 10,
        "dropout_prob": 0.2,
        "armijo_const": 1e-5,
        "grad_norm_clip": 1000,
        "momentum_betas": [0.9, 0.99],
        "eps": 1e-8
    }

    train(train_dataloader, val_dataloader, test_dataloader, T_0, None, losses, metrics, train_config=train_config)
