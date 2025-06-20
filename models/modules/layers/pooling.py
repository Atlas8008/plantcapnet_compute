import torch

from torch import nn


class LogSumExpPoolingLayer(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, x):
        return torch.logsumexp(x, dim=(2, 3), keepdim=self.keepdim)


class MaxPoolingLayer(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, x):
        return torch.amax(x, dim=(2, 3), keepdim=self.keepdim)