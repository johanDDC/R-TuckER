import torch
from torch.optim import Optimizer, Adam

from tucker_riemopt.optimize import get_line_search_tool
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport


class R_TuckEROptimizer(Optimizer):
    def __init__(self, params, model, rank, lr=None, line_search_options=None,
                 scheduler_constructor=None):
        self.line_search = get_line_search_tool(line_search_options)
        self.rank = rank
        self.model = model
        self.lr = self.line_search.alpha_0 if lr is None else lr
        self.line_search = None if lr else self.line_search
        defaults = dict(model=model, rank=rank, lr=self.lr, line_search=self.line_search)
        super().__init__(params, defaults)
        self.regular_optim = Adam(model.parameters(), lr=self.lr)
        if scheduler_constructor:
            self.scheduler = scheduler_constructor(self.regular_optim)

    def loss(self, predictions, targets):
        return self.model.loss(predictions, targets)

    def fit(self, loss_fn, targets):
        x_k = self.model.core
        func = lambda T: loss_fn(T, targets)
        self.riemann_grad = compute_gradient_projection(func, x_k)
        if self.line_search:
            self.lr = self.line_search.line_search(func, x_k, self.riemann_grad, -self.riemann_grad,
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
        x_k -= self.lr * self.riemann_grad
        x_k = x_k.round(self.rank)
        self.model.set_core(x_k)
        self.regular_optim.step()
        del self.riemann_grad

    def scheduler_step(self):
        self.scheduler.step()


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