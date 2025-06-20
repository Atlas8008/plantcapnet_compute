from turtle import forward
import torch
import torch.nn.functional as F

from torch import nn

from .layers import pooling

__all__ = [
    "get_classification_module",
    "ClassificationModule",
]


def get_classification_module(in_feats, out_feats, **kwargs):
    return ClassificationModule(
        in_feats=in_feats,
        out_feats=out_feats,
        **kwargs,
    )


class ClassificationModule(nn.Module):
    def __init__(self, in_feats, out_feats, pooling_method="gap") -> None:
        super().__init__()

        if pooling_method == "gap":
            self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_method == "gmp":
            self.pooling_layer = pooling.MaxPoolingLayer(keepdim=True)
        elif pooling_method == "lse":
            self.pooling_layer = pooling.LogSumExpPoolingLayer(keepdim=True)
        elif pooling_method == "none" or pooling_method is None:
            self.pooling_layer = None
        else:
            raise ValueError("Invalid pooling method: " + pooling_method)

        self.class_conv = nn.Conv2d(
            in_feats,
            out_feats,
            kernel_size=(1, 1),
        )

    def forward(self, x, mode="classification"):
        assert mode in ("classification", "cam")

        if mode == "classification":
            if self.pooling_layer is not None:
                x = self.pooling_layer(x)

        x = self.class_conv(x)

        if mode == "classification":
            x = torch.reshape(x, (x.shape[0], -1))


        return x
