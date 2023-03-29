import torch
import numpy as np

from torch.optim import Optimizer
from tucker_riemopt import Tucker, backend as back
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport


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
        return grad_norm

    @torch.no_grad()
    def __R_step(self, x_k):
        eps = 1e-8
        factor, intermediate_factor = torch.linalg.qr(x_k.factors[0])
        intermediate_tensor = back.einsum("ijk,ai->ajk", x_k.core, intermediate_factor)
        unfolding = back.reshape(intermediate_tensor, (intermediate_tensor.shape[0], -1), order="F")
        u, s, _ = back.svd(unfolding, full_matrices=False)
        eps_svd = eps / np.sqrt(3) * back.sqrt(s @ s)
        cumsum = back.cumsum(back.flip(s))
        cumsum = back.flip(~(cumsum <= eps_svd))
        u = u[:, cumsum]
        core = back.einsum("ijk,ia->ajk", intermediate_tensor, u)
        factor @= u
        factor = factor[:, :self.rank[0]]
        return core[:self.rank[0], :, :], factor

    def __E_step(self, x_k: Tucker, num_iters=1):
        new_factor = torch.randn((x_k.factors[1].shape[0], self.rank[1]), device=x_k.factors[1].device)
        new_factor = torch.linalg.qr(new_factor)[0]
        new_factor.requires_grad_(True)
        cost_fn = lambda U: x_k.k_mode_product(1, U.T).k_mode_product(2, U.T).norm()
        for iter in range(num_iters):
            new_factor.requires_grad_(True)
            cost = cost_fn(new_factor)
            cost.backward()
            grad = new_factor.grad
            with torch.no_grad():
                new_factor = new_factor + grad
                new_factor = torch.linalg.qr(new_factor)[0]
        return new_factor


    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, E, R = self.param_groups[0]["params"]
        x_k = Tucker(W, [R, E, E])

        # self.lr, x_k = self.__armijo(closure, x_k, -self.direction)
        x_k = add(x_k, self.direction, self.lr)
        new_core, new_R = self.__R_step(x_k)
        x_k = Tucker(new_core.detach(), [torch.eye(self.rank[0], device=new_R.device).detach(), x_k.factors[1].detach(), x_k.factors[2].detach()])
        new_E = self.__E_step(x_k, num_iters=100)
        x_k = Tucker(x_k.k_mode_product(1, new_E.T).k_mode_product(2, new_E.T).full().detach(),
                     [new_R.detach(), new_E.detach(), new_E.detach()])

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.factors[0] - R)
        E.data.add_(x_k.factors[1] - E)


def add(x: Tucker, grad: Tucker, alpha):
    rank = x.rank
    temp_core = torch.zeros_like(grad.core, device=grad.core.device)
    temp_core[:rank[0], :rank[1], :rank[2]] = x.core
    sum_core = temp_core + alpha * grad.core
    return Tucker(sum_core, grad.factors)

