import math
import torch

from torch import nn

from .base_time_series_model import TorchTrainableTimeSeriesModel
from .utils import FeatureFuser


class ConvolutionalModel(nn.Module):
    def __init__(self, n_outputs=None, n_feats=64, n_convs=1, kernel_size=3, residual=False, embedding_dimensions=None, normalization=None, fuse_method="concat", depthwise=False) -> None:
        super().__init__()

        if depthwise:
            assert n_feats % n_outputs == 0, "For depthwise convolution the number of features has to be divisible by the number of outputs."

        self.n_outputs = n_outputs
        self.n_feats = n_feats
        self.n_convs = n_convs
        self.kernel_size = kernel_size
        self.feature_fuser = FeatureFuser(
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
        )
        self.depthwise = depthwise

        self.convs = None
        self.res_conv = None
        assert isinstance(residual, (bool, int))
        self.residual = residual if isinstance(residual, bool) else residual

    def _maybe_init(self, input_size):
        if self.convs is None:
            if self.n_outputs is None:
                output_size = input_size
            else:
                output_size = self.n_outputs

            if self.depthwise:
                groups = self.n_outputs
            else:
                groups = 1

            self.convs = self._get_convs(input_size, output_size, groups)
            self.res_conv = nn.Conv1d(input_size, output_size, kernel_size=1, groups=groups)
            print(self)
            # self.conv = nn.Conv1d(
            #     self.hidden_size * d,
            #     output_size,
            #     kernel_size=1
            # )

    def _get_convs(self, input_size, output_size, groups):
        convs = []

        for i in range(self.n_convs):
            convs.append(
                nn.Conv1d(
                    input_size if i == 0 else self.n_feats,
                    output_size if i == self.n_convs - 1 else self.n_feats,
                    kernel_size=self.kernel_size,
                    padding="same",
                    padding_mode="replicate",
                    groups=groups,
                )
            )

            if i < self.n_convs - 1:
                convs.append(nn.ReLU())

        return nn.Sequential(*convs)


    def forward(self, x):
        if not isinstance(self.residual, bool):
            if isinstance(x, (tuple, list)):
                x_orig = x[self.residual]
            else:
                x_orig = x
            use_residual = True
            use_res_conv = False

        if isinstance(x, (tuple, list)):
            x = [torch.swapdims(xi, 1, 2) for xi in x]
        else:
            x = torch.swapdims(x, 1, 2)
        x = self.feature_fuser(x)
        x = torch.swapdims(x, 1, 2)

        self._maybe_init(x.shape[2])

        if isinstance(self.residual, bool):
            x_orig = x
            use_residual = True
            use_res_conv = True

        out = torch.swapdims(x, 1, 2)
        out = self.convs(out)
        out = torch.swapdims(out, 1, 2)

        if use_residual:
            res = x_orig
            if use_res_conv:
                res = torch.swapdims(res, 1, 2)
                res = self.res_conv(res)
                res = torch.swapdims(res, 1, 2)

            out = res + out

        return out


class ConvolutionalTimeSeriesModel(TorchTrainableTimeSeriesModel):
    def __init__(self, n_outputs=None, n_feats=64, n_convs=1, kernel_size=3, residual=False, embedding_dimensions=None, normalization=None, fuse_method="concat", depthwise=False, **kwargs):
        super().__init__(**kwargs)

        self.model = ConvolutionalModel(
            n_outputs=n_outputs,
            n_feats=n_feats,
            n_convs=n_convs,
            kernel_size=kernel_size,
            residual=residual,
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
            depthwise=depthwise,
        )
