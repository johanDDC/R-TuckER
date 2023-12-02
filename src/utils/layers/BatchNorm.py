import torch
import torch.nn as nn

from typing import Any

from torch.nn.modules.batchnorm import _BatchNorm


class _RiemannBatchNorm1d(torch.autograd.Function):
    eps = None
    rank = None
    correction_mask = None
    grad_correction_mask = None

    @staticmethod
    def set_parameters(eps: float, rank: int):
        _RiemannBatchNorm1d.eps = eps
        _RiemannBatchNorm1d.rank = rank
        _RiemannBatchNorm1d.correction_mask = torch.hstack([
            torch.full((rank,), eps),
            torch.ones((rank,))
        ])
        _RiemannBatchNorm1d.grad_correction_mask = torch.hstack([
            torch.ones((rank,)),
            torch.zeros((rank,))
        ])

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        mean = input.mean(dim=0)
        input_mean = input - mean
        var = (input_mean ** 2).mean(dim=0)
        if _RiemannBatchNorm1d.correction_mask.device != input.device:
            _RiemannBatchNorm1d.correction_mask = _RiemannBatchNorm1d.correction_mask.to(input.device)
        std = torch.sqrt(var + _RiemannBatchNorm1d.correction_mask)
        input_hat = input_mean / std
        ctx.save_for_backward(input_hat, std, weight)
        return weight * input_hat + bias, mean, std

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor, mean, std):
        input_hat, std, weight = ctx.saved_tensors
        B, D = grad_output.shape
        if _RiemannBatchNorm1d.grad_correction_mask.device != input_hat.device:
            _RiemannBatchNorm1d.grad_correction_mask = _RiemannBatchNorm1d.grad_correction_mask.to(input_hat.device)

        grad_bias = grad_output.sum(dim=0) * _RiemannBatchNorm1d.grad_correction_mask
        grad_weight = (grad_output * input_hat).sum(dim=0)

        grad_input_hat = grad_output * weight
        grad_input = (1 / (B * std)) * (B * grad_input_hat - grad_input_hat.sum(dim=0) -
                                        input_hat * (grad_input_hat * input_hat).sum(dim=0))
        return grad_input, grad_weight, grad_bias


class RiemannBatchNorm1d(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__(num_features, eps, momentum, affine=True,
                         track_running_stats=True)
        self.running_mean = torch.zeros((2 * num_features,))
        self.running_std = torch.ones((2 * num_features,))

        self.weight = nn.Parameter(torch.ones(2 * num_features,))
        self.bias = nn.Parameter(torch.zeros(2 * num_features,))

        _RiemannBatchNorm1d.set_parameters(eps, num_features)
        self.__bn = _RiemannBatchNorm1d.apply

    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
        if self.running_std.device != x.device:
            self.running_std = self.running_std.to(x.device)
        if self.training:
            result, mean, std = self.__bn(x, self.weight, self.bias)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std.detach()
        else:
            result = (x - self.running_mean[:self.num_features]) / self.running_std[:self.num_features ]
            result = self.weight[:self.num_features] * result + self.bias[:self.num_features]
        return result
