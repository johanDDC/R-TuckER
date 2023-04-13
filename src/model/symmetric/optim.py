import torch
import numpy as np

from torch.optim import Optimizer
from tucker_riemopt import backend as back
from tucker_riemopt.symmetric.tucker import Tucker
from tucker_riemopt.symmetric.riemopt import compute_gradient_projection, vector_transport


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
        # if self.direction:
        #     self.momentum = vector_transport(None, x_k, self.direction)
        riemann_grad = compute_gradient_projection(loss_fn, x_k)
        grad_norm = riemann_grad.norm()
        riemann_grad = 1 / grad_norm * riemann_grad
        # if self.momen tum:
        #     self.direction = self.momentum_beta * self.momentum + (1 - self.momentum_beta) * riemann_grad
        # else:
        self.direction = riemann_grad
        self.loss = loss_fn(x_k)
        return grad_norm

    @torch.no_grad()
    def __R_step(self, x_k: Tucker):
        Q_e, R_e = torch.linalg.qr(x_k.symmetric_factor)
        Q_r, R_r = torch.linalg.qr(x_k.common_factors[0])
        core_wav = torch.einsum("ijk,ai,bj,ck->abc", x_k.core, R_r, R_e, R_e)
        unfolding = torch.flatten(core_wav, 1)
        u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
        u = u[:, :self.rank[0]]
        core_wavwav = torch.einsum("ijk,ai->ajk", core_wav, u.T)
        return Tucker(core_wavwav, [Q_r @ u], x_k.symmetric_modes, Q_e)

    def __E_step(self, x_k: Tucker, num_iters=1):
        new_factor = torch.randn((x_k.symmetric_factor.shape[0], self.rank[1]), device=x_k.symmetric_factor.device)
        new_factor = torch.linalg.qr(new_factor)[0]
        cost_fn = lambda U: x_k.symmetric_modes_product(U.T).norm()
        new_factor.requires_grad_(True)
        # Cayley SGD
        lr, mom_coef, eps, q, s = 0.2, 0.9, 1e-8,0.5, 2
        for iter in range(num_iters):
            cost = cost_fn(new_factor)
            cost.backward()
            eucl_grad = new_factor.grad
            M = mom_coef * M + eucl_grad if iter > 0 else eucl_grad
            # M = eucl_grad
            temp = M - 0.5 * new_factor @ (new_factor.T @ M)
            W = lambda X: temp @ (X.T @ X) - X @ (temp.T @ X)
            # W = lambda X: temp - X @ (temp.T @ X)
            M = W(new_factor)
            # no lr adjustment
            alpha = min(lr, 2 * q / (2 * torch.linalg.norm(eucl_grad, ord="fro") + eps))
            Y = new_factor + alpha * M
            for i in range(s):
                Y = new_factor + alpha / 2 * W(new_factor + Y)
            new_factor = Y.detach()
            new_factor.requires_grad_(True)
        # new_factor = torch.linalg.qr(new_factor)[0]
        return new_factor


    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]
        x_k = Tucker(W, [R], symmetric_modes=[1, 2], symmetric_factor=E)

        # self.lr, x_k = self.__armijo(closure, x_k, -self.direction)
        x_k = x_k - self.lr * self.direction
        # x_k = add(x_k, -self.direction, self.lr)
        x_k = self.__R_step(x_k)
        new_R = x_k.common_factors[0]
        x_k = Tucker(x_k.core, [(new_R.T @ new_R)], x_k.symmetric_modes, x_k.symmetric_factor)
        new_E = self.__E_step(x_k, num_iters=30).detach()
        core = x_k.symmetric_modes_product(new_E.T).full()
        Q_E, R_E = torch.linalg.qr(new_E)
        # print(Q_E.shape, R_E.shape)
        core = back.einsum("ijk,aj,bk->iab", core, R_E, R_E)
        new_E = Q_E

        W.data.add_(core - W)
        R.data.add_(new_R - R)
        E.data.add_(new_E - E)


def add(x: Tucker, grad: Tucker, alpha):
    rank = x.rank
    temp_core = torch.zeros_like(grad.core, device=grad.core.device)
    temp_core[:rank[0], :rank[1], :rank[2]] = x.core
    sum_core = temp_core + alpha * grad.core
    return Tucker(sum_core, grad.common_factors, grad.symmetric_modes, grad.symmetric_factor)

