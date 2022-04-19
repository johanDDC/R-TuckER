import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.optim import Optimizer, Adam

import numpy as np

from tucker_riemopt import Tucker
from tucker_riemopt.optimize import get_line_search_tool
from tucker_riemopt.riemopt import compute_gradient_projection
from tucker_riemopt import backend


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


def compute_metrics(predictions, targets, ks):
    _, idx = torch.sort(predictions, dim=1, descending=True)
    targets_sorted = targets.gather(1, idx)
    ranks = targets_sorted.argmax(1) + 1
    metrics = dict()
    metrics["mrr"] = torch.mean(1 / ranks)
    for k in ks:
        hits = targets_sorted[:, :k].sum(1).float()
        hits[hits > 1] = 1
        metrics["hits_" + str(k)] = torch.mean(hits)
    return metrics


class R_TuckER(torch.nn.Module):
    def __init__(self, data_count, embeddings_dim, rank=None, **kwargs):
        """
        Parameters:
        -----------
        data_count: tuple
            (entities_count, relations_count) --- number of objects of corresponding kind in dataset
        embeddings_dim: tuple
            (entities_dim, relations_dim) --- dimensions of embeddings
        rank: Sequence[int]
            rank of the manifold
        """
        super(R_TuckER, self).__init__()

        self.S = nn.Embedding(data_count[0], embeddings_dim[0])
        self.R = nn.Embedding(data_count[1], embeddings_dim[1])
        self.rank = rank
        self.core = backend.tensor(np.random.uniform(-1, 1, (embeddings_dim[0], embeddings_dim[1], embeddings_dim[0])),
                                 dtype=torch.float)
        self.core = Tucker.full2tuck(self.core)
        self.input_dropout = nn.Dropout(kwargs.get("input_dropout", 0))
        self.hidden_dropout1 = nn.Dropout(kwargs.get("hidden_dropout1", 0))
        self.hidden_dropout2 = nn.Dropout(kwargs.get("hidden_dropout2", 0))
        self.loss = nn.BCELoss()

        self.bn0 = nn.BatchNorm1d(embeddings_dim[0])
        self.bn1 = nn.BatchNorm1d(embeddings_dim[0])

    def init(self):
        xavier_normal_(self.S.weight.data)
        xavier_normal_(self.R.weight.data)

    def set_core(self, T):
        self.core = T

    def forward(self, subject_idx, relation_idx):
        subjects = self.S(subject_idx)
        relations = self.R(relation_idx)

        def forward_core(T, targets):
            x = self.bn0(subjects)
            W = T.k_mode_product(1, relations)
            W = W.k_mode_product(0, x)
            x = W.k_mode_product(2, self.S.weight)
            pred = torch.sigmoid(x.full())
            preds = pred[0, 0, :].reshape(1, -1)
            for i in range(1, pred.shape[0]):
                preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)
            return self.loss(preds, targets)

        x = self.bn0(subjects)
        # x = self.input_dropout(x)
        # x.shape: batch_size x entities_dim
        W = self.core.k_mode_product(1, relations)
        W = W.k_mode_product(0, x)
        x = W.k_mode_product(2, self.S.weight)
        pred = torch.sigmoid(x.full())
        preds = pred[0, 0, :].reshape(1, -1)
        for i in range(1, pred.shape[0]):
            preds = torch.cat([preds, pred[i, i, :].reshape(1, -1)], dim=0)
        return preds, forward_core

        # relations = self.R(relation_idx)
        # W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        # W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        # W_mat = self.hidden_dropout1(W_mat)
        #
        # x = torch.bmm(x, W_mat)
        # x = x.view(-1, e1.size(1))
        # x = self.bn1(x)
        # x = self.hidden_dropout2(x)
        # x = torch.mm(x, self.E.weight.transpose(1,0))
        # pred = torch.sigmoid(x)
        # return pred

    def to(self, device):
        if device == "cuda":
            self.cuda()
        else:
            self.cpu()
        self.set_core(Tucker(self.core.core.to(device), [factor.to(device) for factor in self.core.factors]))


class R_TuckEROptimizer(Optimizer):
    def __init__(self, params, model, rank, lr, line_search_options=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank
        self.model = model
        defaults = dict(model=model, rank=rank, line_search=self.line_search)
        super().__init__(params, defaults)
        self.alpha = self.line_search.alpha_0
        self.regular_optim = Adam(model.parameters(), lr=lr)

    def calc_loss(self, predictions, targets):
        loss = self.model.loss(predictions, targets)
        # loss.backward()
        return loss

    def fit(self, loss_fn, targets):
        x_k = self.model.core
        func = lambda T: loss_fn(T, targets)
        self.riemann_grad = compute_gradient_projection(func, x_k)
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
        x_k = self.model.core
        x_k -= self.alpha * self.riemann_grad
        x_k = x_k.round(self.rank)
        self.model.set_core(x_k)
        self.regular_optim.step()
        del self.riemann_grad


def fit_core(model, loss_fn, targets, line_search_options=None):
    x_k = model.core
    rank = model.rank
    line_search = get_line_search_tool(line_search_options)
    func = lambda T: loss_fn(T, targets)
    riemann_grad = compute_gradient_projection(func, x_k)
    alpha = line_search.line_search(func, x_k, riemann_grad, -riemann_grad, rank,
                                    2 * line_search.alpha_0)
    x_k -= alpha * riemann_grad
    x_k = x_k.round(rank)
    return x_k
