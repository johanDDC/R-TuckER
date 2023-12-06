from typing import Callable, Union

import torch

from torch.optim import Optimizer
from tucker_riemopt import Tucker
from tucker_riemopt import TuckerRiemannian


class RGD(Optimizer):
    def __init__(self, params, rank, max_lr):
        self.rank = rank
        self.max_lr = max_lr
        self.lr = max_lr

        defaults = dict(rank=rank, max_lr=self.max_lr, lr=self.lr)
        super().__init__(params, defaults)

        self.direction = None
        self.loss = None

    def fit(self, loss_fn: Callable[[Tucker], float], x_k: Tucker,
            normalize_grad: Union[float, "False"] = 1.):
        """Computes the Riemannian gradient of `loss_fn` at point `x_k`.

            :param loss_fn: smooth scalar-valued loss function
            :param x_k: current solution approximation
            :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
                be normalized to `normalize_grad`.
            :return: Frobenius norm of the Riemannian gradient.
        """
        rgrad, self.loss = TuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        normalize_grad = rgrad_norm if not normalize_grad else normalize_grad
        normalizer = 1 / rgrad_norm * normalize_grad

        self.direction = normalizer * rgrad
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

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + TuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


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
        self._rank_slices = tuple([slice(0, self.rank[i], None) for i in range(len(self.rank))])

    def fit(self, loss_fn: Callable[[Tucker], float], x_k: Tucker,
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
            self.momentum = TuckerRiemannian.project(x_k, self.direction)
        else:
            self.momentum = TuckerRiemannian.TangentVector(x_k, torch.zeros_like(x_k.core))
        rgrad, self.loss = TuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        normalize_grad = rgrad_norm if not normalize_grad else normalize_grad
        self.direction = (1 / rgrad_norm * normalize_grad) * rgrad + self.momentum_beta * self.momentum
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, S, R, O = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + TuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)
        self.direction = self.direction.construct()

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.factors[0] - R)
        S.data.add_(x_k.factors[1] - S)
        O.data.add_(x_k.factors[2] - O)


class RiemannianAdam(RGD):
    def __init__(self, params, rank, max_lr, betas=(0.9, 0.999), eps=1e-8, step_velocity=1):
        super().__init__(params, rank, max_lr)
        self.betas = betas
        self.eps = eps
        self.step_velocity = step_velocity
        
        self.momentum = None
        self.second_momentum = torch.zeros(1, device="cuda")
        
        self.step_t = 1

    def fit(self, loss_fn: Callable[[Tucker], float], x_k: Tucker,
            normalize_grad: Union[float, "False"] = 1.):
        """
        Computes the Riemannian gradient of `loss_fn` at point `x_k`.

        :param loss_fn: smooth scalar-valued loss function
        :param x_k: current solution approximation
        :param normalize_grad: Can be `False` or float. If `False`, the Riemannian gradient will not be normalized. Otherwise, gradient will
         be normalized to `normalize_grad`.
        :return: Frobenius norm of the Riemannian gradient.
        """
        rgrad, self.loss = TuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        if self.momentum is not None:
            self.momentum = TuckerRiemannian.project(x_k, self.momentum.construct())
            self.momentum = self.betas[0] * self.momentum + (1 - self.betas[0]) * rgrad
        else:
            self.momentum = (1 - self.betas[0]) * rgrad
        self.second_momentum = self.betas[1] * self.second_momentum + (1 - self.betas[1]) * rgrad_norm ** 2
        second_momentum_corrected = self.second_momentum / (1 - self.betas[1] ** (self.step_t // self.step_velocity + 1))
        bias_correction_ratio = (1 - self.betas[0] ** (self.step_t // self.step_velocity + 1)) * torch.sqrt(
            second_momentum_corrected
        ) + self.eps
        self.direction = (1 / bias_correction_ratio) * self.momentum
        return rgrad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, S, R, O = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + TuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)
        self.direction = self.direction.construct()

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.factors[0] - R)
        S.data.add_(x_k.factors[1] - S)
        O.data.add_(x_k.factors[2] - O)
        
        self.step_t += 1
