import torch
import torch.nn as nn

from typing import Any


class _RiemannFactorDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, proba: float):
        # input: B x 2R, where R is corresponding rank
        mask = torch.full((input.shape[0], input.shape[1] // 2), 1 - proba,
                          device=input.device, requires_grad=False)
        mask = torch.bernoulli(mask) / (1 - proba)
        mask = torch.hstack([mask, torch.ones_like(mask, device=mask.device)])
        ctx.save_for_backward(mask)
        return input * mask

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


class RiemannFactorDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.__dropout = _RiemannFactorDropout.apply

    def forward(self, x):
        if self.training:
            return self.__dropout(x, self.p)
        else:
            return x


class _CoreDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, proba: float):
        # input: 2R x 2R x 2R, where R is corresponding component of multilinear rank
        mask = torch.zeros_like(input, device=input.device, requires_grad=False)
        r = [x // 2 for x in input.shape]
        mask[:r[0], :r[1], :r[2]] = proba
        mask = torch.bernoulli(1 - mask)
        ctx.save_for_backward(mask)
        ctx.proba = proba
        return (input * mask) / (1 - proba)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        mask, = ctx.saved_tensors
        return (grad_output * mask) / (1 - ctx.proba), None


class CoreDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.__dropout = _CoreDropout.apply

    def forward(self, x):
        if self.training:
            return self.__dropout(x, self.p)
        else:
            return x