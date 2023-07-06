from typing import Callable, Union

import torch

from torch.optim import Optimizer
from tucker_riemopt import SFTucker
from tucker_riemopt import SFTuckerRiemannian


class RSGDwithMomentum(Optimizer):
    def __init__(self, params, rank, max_lr, momentum_beta=0.9):
        self.rank = rank
        self.max_lr = max_lr
        self.lr = max_lr
        self.momentum_beta = momentum_beta

        defaults = dict(rank=rank, max_lr=self.max_lr, lr=self.lr, momentum_beta=self.momentum_beta)
        super().__init__(params, defaults)

        self.momentum = None
        self.direction = None
        self.loss = None
        self._rank_slices = tuple([slice(0, self.rank[i], None) for i in range(len(self.rank))])

    def fit(self, loss_fn: Callable[[SFTucker], float], x_k: SFTuckerRiemannian,
            normalize_grad: Union[float, "False"] = 1.):
        """
        Computes the Riemannian gradient of `loss_fn` at point `x_k`.

        :param loss_fn: smooth scalar-valued loss function
        :param x_k: current solution approximation
        :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
         be normalized to `normalize_grad`.
        :return: Frobenius norm of the Riemannian gradient.
        """
        if self.direction is not None:
            self.momentum = SFTuckerRiemannian.project(x_k, self.direction)
        else:
            self.momentum = SFTuckerRiemannian.TangentVector(x_k, torch.zeros_like(x_k.core))
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.construct().norm(qr_based=True).detach()
        normalize_grad = rgrad_norm if not normalize_grad else normalize_grad
        self.direction = rgrad.linear_comb(1 / rgrad_norm * normalize_grad, self.momentum_beta,
                                           self.momentum)
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.linear_comb(-self.param_groups[0]["lr"]).construct()
        x_k = x_k.round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)

        self.direction = self.direction.construct()
