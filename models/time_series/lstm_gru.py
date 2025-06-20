import torch

from torch import nn

from .base_time_series_model import TorchTrainableTimeSeriesModel
from .utils import FeatureFuser


class RNNModel(nn.Module):
    def __init__(self, rnn_kind, n_outputs=None, num_layers=1, hidden_size=64, bidirectional=False, n_convs=1, embedding_dimensions=None, normalization=None, fuse_method="concat", residual=False) -> None:
        super().__init__()

        assert rnn_kind in ("lstm", "gru")

        self.n_outputs = n_outputs
        self.rnn_kind = rnn_kind
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_convs = n_convs
        self.feature_fuser = FeatureFuser(
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
        )
        assert isinstance(residual, (bool, int))
        self.residual = residual if isinstance(residual, bool) else residual

        self.rnn = None
        self.conv = None
        self.res_conv = None

    def _maybe_init_rnn(self, input_size):
        if self.rnn is None:
            d = 2 if self.bidirectional else 1
            if self.n_outputs is None:
                output_size = input_size
            else:
                output_size = self.n_outputs

            if self.rnn_kind == "lstm":
                rnn_fun = nn.LSTM
            elif self.rnn_kind == "gru":
                rnn_fun = nn.GRU

            self.rnn = rnn_fun(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.conv = self._get_convs(d, output_size)
            self.res_conv = nn.Conv1d(input_size, output_size, kernel_size=1)

            print(self)
            # self.conv = nn.Conv1d(
            #     self.hidden_size * d,
            #     output_size,
            #     kernel_size=1
            # )

    def _get_convs(self, d, output_size):
        convs = []

        for i in range(self.n_convs):
            convs.append(
                nn.Conv1d(
                    self.hidden_size * d,
                    output_size if i == self.n_convs - 1 else self.hidden_size * d,
                    kernel_size=1
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

        self._maybe_init_rnn(x.shape[2])

        if isinstance(self.residual, bool):
            x_orig = x
            use_residual = True
            use_res_conv = True

        d = 2 if self.bidirectional else 1

        h0 = torch.zeros((self.num_layers * d, x.shape[0], self.hidden_size))

        if self.rnn_kind == "lstm":
            c0 = torch.zeros((self.num_layers * d, x.shape[0], self.hidden_size))
            out, _ = self.rnn(x, (h0, c0))
        elif self.rnn_kind == "gru":
            out, _ = self.rnn(x, h0)

        out = torch.swapdims(out, 1, 2)
        out = self.conv(out)
        out = torch.swapdims(out, 1, 2)

        if use_residual:
            res = x_orig
            if use_res_conv:
                res = torch.swapdims(res, 1, 2)
                res = self.res_conv(res)
                res = torch.swapdims(res, 1, 2)

            out = res + out

        return out


class RNNTimeSeriesModel(TorchTrainableTimeSeriesModel):
    def __init__(self, rnn_kind, n_outputs=None, num_layers=1, bidirectional=False, hidden_units=64, n_convs=1, embedding_dimensions=None, normalization=None, fuse_method="concat", residual=False, **kwargs):
        super().__init__(**kwargs)

        self.model = RNNModel(
            rnn_kind=rnn_kind,
            n_outputs=n_outputs,
            num_layers=num_layers,
            bidirectional=bidirectional,
            hidden_size=hidden_units,
            n_convs=n_convs,
            residual=residual,
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
        )
