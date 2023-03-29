import torch
import torch.nn as nn


class DropoutLayer(nn.Module):
    def __init__(self, p, mode_2d=True):
        super().__init__()
        self.p = 1 - p
        self.mode_2d = mode_2d
        self.backward_mask = None

    def forward(self, input, forward_pass=True):
        if self.training:
            if forward_pass:
                mask = torch.full_like(input, self.p)
                self.mask = mask = torch.bernoulli(mask)
                if self.backward_mask is None:
                    if self.mode_2d:
                        self.backward_mask = torch.ones_like(input, dtype=torch.float32)
                    else:
                        self.backward_mask = torch.ones((input.shape[0], 2 * input.shape[1], 2 * input.shape[2]),
                                                        device=input.device, dtype=torch.float32)
            else:
                if self.mode_2d:
                    mask = torch.cat([self.mask, self.backward_mask], dim=1)
                else:
                    mask = self.backward_mask
                    mask[:, :input.shape[1] // 2, :input.shape[2] // 2] = self.mask

            return (input * mask) / self.p
        else:
            return input


class CoreDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = 1 - p
        self.backward_mask = None
        self.rank = None

    def forward(self, x, forward_pass=True):
        if self.training:
            if forward_pass:
                mask = torch.full_like(x, self.p)
                self.mask = mask = torch.bernoulli(mask)
                if self.rank is None:
                    self.rank = x.shape
                if self.backward_mask is None:
                    self.backward_mask = torch.ones((2 * self.rank[0], 2 * self.rank[1], 2 * self.rank[2]),
                                                    device=x.device, dtype=torch.float32)
            else:
                mask = self.backward_mask
                mask[:self.rank[0], :self.rank[1], :self.rank[2]] = self.mask
            return (x * mask) / self.p
        else:
            return x
