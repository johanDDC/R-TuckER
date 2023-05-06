import torch

from torch.optim import Optimizer
from tucker_riemopt.symmetric.tucker import Tucker as SymTucker
from tucker_riemopt.symmetric.riemopt import compute_gradient_projection, vector_transport, group_cores


class SGDmomentum(Optimizer):
    def __init__(self, params, rank, max_lr, momentum_beta=0.9, armijo_slope=1e-4, armijo_increase=0.5,
                 armijo_decrease=0.5, armijo_iters=20):
        self.rank = rank
        self.max_lr = max_lr
        self.lr = max_lr
        self.momentum_beta = momentum_beta

        self.armijo_slope = armijo_slope
        self.armijo_increase = armijo_increase
        self.armijo_decrease = armijo_decrease
        self.armijo_iters = armijo_iters

        defaults = dict(rank=rank, max_lr=self.max_lr, lr=self.lr, momentum_beta=self.momentum_beta,
                        armijo_slope=self.armijo_slope, armijo_increase=self.armijo_increase,
                        armojo_decrease=self.armijo_decrease)
        super().__init__(params, defaults)

        self.momentum = None
        self.direction = None
        self.loss = None
        self._rank_slices = tuple([slice(0, self.rank[i], None) for i in range(len(self.rank))])

    def fit(self, loss_fn, x_k):
        if self.direction is not None:
            self.momentum, _ = vector_transport(None, x_k, self.direction)
        riemann_grad, self.loss = compute_gradient_projection(loss_fn, x_k)
        grad_norm = riemann_grad.norm(qr_based=True).detach()
        if self.momentum is not None:
            self.direction = self._add_momentum(x_k, riemann_grad, self.momentum,
                                                self.momentum_beta, grad_norm)
        else:
            riemann_grad = 1 / grad_norm * riemann_grad
            self.direction = riemann_grad
        return grad_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]
        x_k = SymTucker(W, [R], num_symmetric_modes=2, symmetric_factor=E)

        # self.lr, x_k = self.__armijo(closure, x_k, -self.direction)
        x_k = self._add(x_k, -self.direction, self.param_groups[0]["lr"])
        x_k = x_k.round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.common_factors[0] - R)
        E.data.add_(x_k.symmetric_factor - E)

    def _add_momentum(self, x_k: SymTucker, grad: SymTucker, momentum: SymTucker, beta: float, grad_norm: float):
        normalize = 1 / grad_norm
        dG1 = grad.core[self._rank_slices]
        dG2 = momentum.core[self._rank_slices]
        V_sum = []
        for i in range(x_k.ndim - x_k.num_symmetric_modes):
            dV1 = grad.common_factors[i][:, self.rank[i]:]
            dV2 = momentum.common_factors[i][:, self.rank[i]:]
            dV_sum = normalize * dV1 + beta * dV2
            V_sum.append(torch.hstack([x_k.common_factors[i], dV_sum]))
        dU1 = grad.symmetric_factor[:, self.rank[-1]:]
        dU2 = momentum.symmetric_factor[:, self.rank[-1]:]
        U_sum = torch.hstack([x_k.symmetric_factor, normalize * dU1 + beta * dU2])
        G_sum = group_cores(normalize * dG1 + beta * dG2, x_k.core)
        return SymTucker(G_sum, V_sum, 2, U_sum)

    def _add(self, x: SymTucker, direction: SymTucker, alpha):
        temp_core = torch.zeros_like(direction.core, device=direction.core.device)
        temp_core[self._rank_slices] = x.core
        sum_core = temp_core + alpha * direction.core
        return SymTucker(sum_core, direction.common_factors, direction.num_symmetric_modes, direction.symmetric_factor)
