import torch
from torch.optim import Optimizer, Adam

from tucker_riemopt import Tucker
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport


class SGDmomentum(Optimizer):
    def __init__(self, params, rank, lr, momentum_beta=0.9, armijo_slope=1e-4, armijo_increase=0.5,
                 armojo_decrease=0.5, armijo_iters = 20):
        self.rank = rank
        self.lr = lr
        self.momentum_beta = momentum_beta
        self.armijo_slope = armijo_slope
        self.armijo_increase = armijo_increase
        self.armijo_decrease = armojo_decrease
        self.armijo_iters = armijo_iters
        defaults = dict(rank=rank, lr=self.lr, momentum_beta=self.momentum_beta,
                        armijo_slope=self.armijo_slope, armijo_increase=self.armijo_increase,
                        armojo_decrease=self.armijo_decrease)
        super().__init__(params, defaults)
        self.momentum = None
        self.direction = None

    def fit(self, loss_fn, x_k):
        if self.direction:
            self.momentum = vector_transport(None, x_k, self.direction)
        riemann_grad = compute_gradient_projection(loss_fn, x_k)
        grad_norm = riemann_grad.norm()
        riemann_grad = 1 / grad_norm * riemann_grad
        if self.momentum:
            # self.direction = -self.lr * riemann_grad + self.momentum_beta * self.momentum
            self.direction = self.momentum_beta * self.momentum + (1 - self.momentum_beta) * riemann_grad
        else:
            self.direction = riemann_grad
        return  grad_norm

    def __armijo(self, func, x_k, direction):
        alpha = self.lr
        phi_0 = func(x_k)
        last_phi = func(x_k + alpha * direction)
        best = (alpha, last_phi)
        satisfied = phi_0 - last_phi >= alpha * self.armijo_slope

        while not satisfied:
            alpha /= self.armijo_increase
            phi_alpha = func(x_k + alpha * direction)
            if phi_alpha > last_phi:
                alpha *= self.armijo_increase
                satisfied = phi_0 - last_phi >= alpha * self.armijo_slope
                break
            if phi_alpha < best[1]:
                best = (alpha, phi_alpha)
            last_phi = phi_alpha
            satisfied = phi_0 - last_phi >= alpha * self.armijo_slope

        while not satisfied:
            alpha *= self.armijo_decrease
            phi_alpha = func(x_k + alpha * direction)
            if phi_alpha > last_phi:
                alpha /= self.armijo_decrease
                break
            if phi_alpha < best[1]:
                best = (alpha, phi_alpha)
            last_phi = phi_alpha
            satisfied = phi_0 - last_phi >= alpha * self.armijo_slope

        return best[0]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Parameters:
        -----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        W, S, R, O = self.param_groups[0]["params"]
        x_k = Tucker(W, [R, S, O])

        self.lr = self.__armijo(closure, x_k, -self.direction)
        x_k -= self.lr * self.direction
        x_k = x_k.round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.factors[0] - R)
        S.data.add_(x_k.factors[1] - S)
        O.data.add_(x_k.factors[2] - O)

    def scheduler_step(self):
        self.scheduler.step()
