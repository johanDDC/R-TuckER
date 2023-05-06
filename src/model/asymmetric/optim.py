import torch

from torch.optim import Optimizer
from tucker_riemopt import Tucker
from tucker_riemopt.riemopt import compute_gradient_projection, vector_transport, group_cores


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
        if self.direction:
            self.momentum, _ = vector_transport(None, x_k, self.direction)
        riemann_grad, self.loss = compute_gradient_projection(loss_fn, x_k)
        grad_norm = riemann_grad.norm()
        riemann_grad = 1 / grad_norm * riemann_grad
        if self.momentum:
            self.direction = self._add_momentum(x_k, riemann_grad, self.momentum, self.momentum_beta, 1 / grad_norm)
        else:
            self.direction = riemann_grad
        return grad_norm

    def __armijo(self, func, x_k: Tucker, direction: Tucker):
        alpha = 2 * self.lr
        phi_0 = func(x_k)
        grad_norm = direction.norm(qr_based=False)  # set qr_based=True
        iter_num = 0

        retract_alpha = add_and_retract(x_k, direction)
        new_point = retract_alpha(alpha)
        last_phi = func(new_point)
        satisfied = phi_0 - last_phi >= alpha * self.armijo_slope * grad_norm

        while not satisfied and iter_num < self.armijo_iters:
            alpha *= self.armijo_decrease
            new_point = retract_alpha(alpha)
            last_phi = func(new_point)
            iter_num += 1
            satisfied = phi_0 - last_phi >= alpha * self.armijo_slope * grad_norm

        return alpha, new_point

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

        # self.lr, x_k = self.__armijo(closure, x_k, -self.direction)
        x_k = add_and_retract(x_k, -self.direction)(self.param_groups[0]["lr"])
        # x_k = self.add_and_retract_momentum(x_k, -self.direction)(self.param_groups[0]["lr"])

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.factors[0] - R)
        S.data.add_(x_k.factors[1] - S)
        O.data.add_(x_k.factors[2] - O)

    def _add_momentum(self, x_k, grad: Tucker, momentum: Tucker, beta: float, grad_norm: float):
        normalize = 1 / grad_norm
        dG1 = grad.core[:self.rank[0], :self.rank[1], :self.rank[2]]
        dR1 = grad.factors[0][:, self.rank[0]:]
        dS1 = grad.factors[1][:, self.rank[1]:]
        dO1 = grad.factors[2][:, self.rank[2]:]
        dG2 = momentum.core[:self.rank[0], :self.rank[1], :self.rank[2]]
        dR2 = momentum.factors[0][:, self.rank[0]:]
        dS2 = momentum.factors[1][:, self.rank[1]:]
        dO2 = momentum.factors[2][:, self.rank[2]:]
        sum_G = group_cores(normalize * dG1 + beta * dG2, x_k.core)
        sum_R = torch.hstack([x_k.common_factors[0], normalize * dR1 + beta * dR2])
        sum_S = torch.hstack([x_k.symmetric_factor, normalize * dS1 + beta * dS2])
        sum_O = torch.hstack([x_k.symmetric_factor, normalize * dO1 + beta * dO2])
        return Tucker(sum_G, [sum_R, sum_S, sum_O])

def add_and_retract(x: Tucker, grad: Tucker):
    rank = x.rank
    Qs = [None] * x.ndim
    Rs = [None] * x.ndim
    for i in range(x.ndim):
        Qs[i], Rs[i] = torch.linalg.qr(grad.factors[i])

    def f(alpha):
        temp_core = torch.zeros_like(grad.core, device=grad.core.device)
        temp_core[:rank[0], :rank[1], :rank[2]] = x.core
        sum_core = temp_core + alpha * grad.core
        inner_tensor = Tucker(sum_core, Rs)
        inner_tensor = Tucker.full2tuck(inner_tensor.full(), eps=1e-8)
        factors = [None] * x.ndim
        for i in range(x.ndim):
            factors[i] = Qs[i] @ inner_tensor.factors[i]
            factors[i] = factors[i][:, :rank[i]]
        return Tucker(inner_tensor.core[:rank[0], :rank[1], :rank[2]], factors)

    return f
