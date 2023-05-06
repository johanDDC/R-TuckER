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
    def retraction(self, T: SymTucker):
        Q_e, R_e = torch.linalg.qr(T.symmetric_factor)
        Q_r, R_r = torch.linalg.qr(T.common_factors[0])
        core_wav = torch.einsum("ijk,ai,bj,ck->abc", T.core, R_r, R_e, R_e)
        unfolding = torch.flatten(core_wav, 1)
        u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
        u = u[:, :self.rank[0]]
        R = Q_r @ u
        G_2 = torch.flatten(torch.permute(core_wav, (1, 0, 2)), 1)
        G_3 = torch.flatten(torch.permute(core_wav, (2, 0, 1)), 1)
        G_concat = torch.hstack([G_2, G_3])
        u, _, _ = torch.linalg.svd(G_concat, full_matrices=False)
        u = u[:, :self.rank[1]]
        E = Q_e @ u
        core = T.k_mode_product(0, R.T).symmetric_modes_product(E.T).full()
        return SymTucker(core, [R], 2, E)

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
        x_k = self._add(x_k, -self.direction, self.lr)
        x_k = self.retraction(x_k)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.common_factors[0] - R)
        E.data.add_(x_k.symmetric_factor - E)

    def _add_momentum(self, x_k: SymTucker, grad: SymTucker, momentum: SymTucker, beta: float, grad_norm: float):
        normalize = 1 / grad_norm
        dG1 = grad.core[:self.rank[0], :self.rank[1], :self.rank[2]]
        dV1 = grad.common_factors[0][:, self.rank[0]:]
        dU1 = grad.symmetric_factor[:, self.rank[1]:]
        dG2 = momentum.core[:self.rank[0], :self.rank[1], :self.rank[2]]
        dV2 = momentum.common_factors[0][:, self.rank[0]:]
        dU2 = momentum.symmetric_factor[:, self.rank[1]:]
        sum_G = group_cores(normalize * dG1 + beta * dG2, x_k.core)
        sum_V = torch.hstack([x_k.common_factors[0], normalize * dV1 + beta * dV2])
        sum_U = torch.hstack([x_k.symmetric_factor, normalize * dU1 + beta * dU2])
        return SymTucker(sum_G, [sum_V], 2, sum_U)

    def _add(self, x: SymTucker, direction: SymTucker, alpha):
        temp_core = torch.zeros_like(direction.core, device=direction.core.device)
        temp_core[:self.rank[0], :self.rank[1], :self.rank[2]] = x.core
        sum_core = temp_core + alpha * direction.core
        return SymTucker(sum_core, direction.common_factors, direction.num_symmetric_modes, direction.symmetric_factor)
