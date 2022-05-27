import jax
import jax.numpy as jnp

import numpy as np
import torch
from torch.optim import Optimizer, Adam
from tucker_riemopt import Tucker
from copy import deepcopy

from tucker_riemopt.optimize import get_line_search_tool
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport

from collections import deque


class R_TuckEROptimizer(Optimizer):
    def __init__(self, params, model, rank, lr, line_search_options=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank
        self.model = model
        defaults = dict(model=model, rank=rank, lr=lr, line_search=self.line_search)
        super().__init__(params, defaults)
        self.alpha = lr
        self.lr = None
        if lr == "Optimal":
            self.lr = True
            self.line_search = get_line_search_tool(line_search_options)
            self.alpha = self.line_search.alpha_0


    def loss(self, predictions, targets):
        loss = self.model.loss(predictions, targets)
        return loss

    def fit(self, loss_fn, targets):
        x_k = self.model.tucker
        func = lambda T: loss_fn(T, targets)
        self.riemann_grad = compute_gradient_projection(func, x_k)
        if self.lr:
            self.alpha = self.line_search.line_search(func, x_k, self.riemann_grad, -self.riemann_grad,
                                                      self.rank, 1)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        x_k = self.model.tucker
        x_k -= self.alpha * self.riemann_grad
        x_k = x_k.round(self.rank)
        # x_k.factors[2] = x_k.factors[0]
        self.model.tucker = x_k
        del self.riemann_grad


class RSVRG(Optimizer):
    def __init__(self, params, model, rank, lr, len_batches, memory=10, line_search_options=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank if type(rank) is not int else [rank, rank, rank]
        self.model = model
        self.lr = lr
        self.memory = memory
        defaults = dict(model=model, rank=rank, lr=lr, memory=memory, line_search=self.line_search)
        super().__init__(params, defaults)
        self.rieman_grad = None
        self.rieman_grad_next = None
        self.len_batches = len_batches
        self.idx = np.random.choice(np.arange(0, self.len_batches), self.memory, False)
        self.idx = torch.tensor(self.idx)
        self.idx, _ = torch.sort(self.idx)
        self.fit_count = 0
        self.saved_grads = [[], []]
        self.x_k = model.tucker

    def loss(self, predictions, targets):
        return self.model.loss(predictions, targets)

    def _iter(self, loss_fn, targets, retain_graph=False):
        func = lambda T: loss_fn(T, targets)
        # x_k = deepcopy(self.model.tucker)
        x_k = self.model.tucker

        grad = compute_gradient_projection(func, x_k)
        self.rieman_grad = grad if self.rieman_grad is None else \
            self.rieman_grad + grad
        self.rieman_grad = Tucker(self.rieman_grad.core.detach(),
                                  [factor.detach() for factor in self.rieman_grad.factors])
        if self.fit_count > 1:
            self.rieman_grad = self.rieman_grad.round(self.rank * 2)
        if (self.fit_count - 1) in self.idx:
            self.saved_grads[1].append(grad)

    def fit(self, loss_fn, targets):
        self.fit_count += 1
        if self.rieman_grad_next is None:
            self._iter(loss_fn, targets)
        else:
            if (self.fit_count - 1) in self.idx:
                func = lambda T: loss_fn(T, targets)
                x_k = deepcopy(self.model.tucker)

                grad = compute_gradient_projection(func, x_k, retain_graph=True)
                self.rieman_grad = grad if self.rieman_grad is None else \
                    self.rieman_grad + grad
                if self.fit_count > 1:
                    self.rieman_grad = self.rieman_grad.round(self.rank * 2)
                if (self.fit_count - 1) in self.idx:
                    self.saved_grads[1].append(grad)

                x_k_copy = deepcopy(self.x_k)
                v = compute_gradient_projection(func, x_k_copy, retain_graph=True) - \
                    vector_transport(None, x_k_copy, self.saved_grads[0][
                        (self.idx == (self.fit_count - 1)).nonzero().item()] - self.rieman_grad_next, retain_graph=True)
                alpha = self.line_search.line_search(func, x_k, v, -v,
                                                     self.rank, self.lr)
                self.x_k -= alpha * v
                self.x_k = self.x_k.round(self.rank)
            else:
                self._iter(loss_fn, targets)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:ещё о
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        if self.rieman_grad_next is not None:
            # self.x_k.factors[2] = self.x_k.factors[0]
            self.model.tucker = Tucker(self.x_k.core.detach(), [factor.detach() for factor in self.x_k.factors])
            del self.x_k
            self.x_k = self.model.tucker

            self.idx = np.random.choice(np.arange(0, self.len_batches), self.memory, False)
            self.idx = torch.tensor(self.idx)
            self.idx, _ = torch.sort(self.idx)

        self.saved_grads[0], self.saved_grads[1] = self.saved_grads[1], []
        self.rieman_grad = (1 / self.fit_count) * self.rieman_grad
        self.rieman_grad_next = self.rieman_grad
        self.rieman_grad = None
        self.fit_count = 0




def MRR_metrics(predictions, targets):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    mrr = torch.mean(1 / ranks)
    return mrr


def hits_k_metrics(predictions, targets, k = 1):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    hits = targets_sorted[:, :k].sum(1).float()
    hits[hits > 1] = 1
    return torch.mean(hits)


def compute_metrics(predictions, targets, ks, accum=None):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    metrics = dict(mrr=0) if not accum else accum
    metrics["mrr"] += torch.sum(1 / ranks)
    for k in ks:
        if not accum:
            metrics[f"hits_{k}"] = 0
        hits = targets_sorted[:, :k].sum(1).float()
        hits[hits > 1] = 1
        metrics["hits_" + str(k)] += torch.sum(hits)
    return metrics


def filter_predictions(predictions, targets, filter):
    # removing all scores of actual true triplets in predictions but one we interested in.
    # useful for computing filtered MRR and Hits@k
    interest_prediction_vals = predictions.gather(1, filter)
    predictions[targets == 1] = 0
    targets[targets == 1] = 0
    return predictions.scatter_(1, filter, interest_prediction_vals), \
            targets.scatter_(1, filter, torch.ones(interest_prediction_vals.shape, device=targets.device))


def BCELoss(x, y, reduction="mean"):
    loss = y * jnp.log(x) + (1 - y) * jnp.log(1 - x)
    loss *= -1

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class SparseMatrix:
    def __init__(self, inds, vals, shape):
        self.inds = inds
        self.vals = vals
        self.shape = shape

    def __matmul__(self, other):
        rows, cols = self.inds
        in_ = other.take(cols, axis=0)
        prod = in_ * self.vals[:, None]
        res = jax.ops.segment_sum(prod, rows, self.shape[0])
        return res
