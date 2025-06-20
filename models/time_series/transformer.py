import math
import torch

from torch import nn

from .base_time_series_model import TorchTrainableTimeSeriesModel
from .utils import FeatureFuser


class SinCosEncoding(nn.Module):
    @torch.no_grad()
    def __init__(self, d, n=10000) -> None:
        super().__init__()

        self.d = d

        self.div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(float(n)) / d))[None, :, None]

    @torch.no_grad()
    def forward(self, positions):
        pe = torch.zeros((1, self.d, positions.shape[2]))

        pe[:, 0::2, :] = torch.sin(positions * self.div_term)
        pe[:, 1::2, :] = torch.cos(positions * self.div_term)

        return pe


class PositionalEncoding(nn.Module):
    def __init__(self, kind, out_feats=None) -> None:
        super().__init__()

        self.kind = kind

        self.pos_encoding_type = None

        self.out_feats = out_feats
        self.embedding = None
        self.pos_embedding = None

    def _maybe_initialize(self, x):
        if self.pos_embedding is None:
            if self.kind in ("rel", "abs"):
                d = x.shape[2]

                self.out_feats = self.out_feats or d
                self.embedding = nn.Conv1d(d, d, 1)
                self.pos_embedding = nn.Conv1d(1, d, 1)
                if self.kind == "rel":
                    self.pos_encoding_type = "relative"
                elif self.kind == "abs":
                    self.pos_encoding_type = "absolute"
            elif self.kind in ("relsc", "abssc"):
                d = x.shape[2]

                self.out_feats = self.out_feats or d
                self.embedding = nn.Conv1d(d, self.out_feats, 1)
                self.pos_embedding = SinCosEncoding(self.out_feats)
                if self.kind == "relsc":
                    self.pos_encoding_type = "relative"
                elif self.kind == "abssc":
                    self.pos_encoding_type = "absolute"
            else:
                raise ValueError("Invalid positional encoding type: " + self.kind)

    def _get_positions(self, x):
        if not self.pos_encoding_type:
            raise ValueError("No positional embedding type was provided.")

        if self.pos_encoding_type == "relative":
            encodings = torch.arange(0, x.shape[2]) / x.shape[2]
            encodings = encodings[None, None]
        elif self.pos_encoding_type == "absolute":
            encodings = torch.arange(0, x.shape[2]) / 1.0
            encodings = encodings[None, None]
        else:
            raise NotImplementedError("Positional encoding type is not implemented: " + self.pos_encoding_type)

        return encodings


    def forward(self, x):
        self._maybe_initialize(x)

        x = torch.swapdims(x, 1, 2)
        x = self.embedding(x)
        embedded_pos_encodings = self.pos_embedding(self._get_positions(x))
        x = x + embedded_pos_encodings
        x = torch.swapdims(x, 1, 2)

        return x


class TransformerModel(nn.Module):
    def __init__(self, n_outputs=None, n_heads=1, n_feats=2048, n_convs=1, positional_encoding=None, pos_feats=None, dropout=0.0, residual=False, embedding_dimensions=None, normalization=None, fuse_method="concat") -> None:
        super().__init__()

        self.n_outputs = n_outputs
        self.n_feats = n_feats
        self.n_heads = n_heads
        self.n_convs = n_convs

        self.pos_feats = pos_feats
        self.encoder = None
        self.conv = None
        self.res_conv = None
        self.dropout = nn.Dropout(p=dropout)
        assert isinstance(residual, (bool, int))
        self.residual = residual if isinstance(residual, bool) else residual
        self.feature_fuser = FeatureFuser(
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
        )

        if positional_encoding:
            self.positional_encoding = PositionalEncoding(positional_encoding, self.pos_feats)
        else:
            self.positional_encoding = None

    def _maybe_init(self, input_size):
        if self.encoder is None:
            self.pos_feats = self.pos_feats or input_size
            if self.n_outputs is None:
                output_size = input_size
            else:
                output_size = self.n_outputs

            self.encoder = nn.TransformerEncoderLayer(
                d_model=self.pos_feats,
                nhead=self.n_heads,
                dim_feedforward=self.n_feats,
                batch_first=True,
            )
            self.conv = self._get_convs(self.pos_feats, output_size)
            self.res_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
            print(self)

    def _get_convs(self, input_size, output_size):
        convs = []

        for i in range(self.n_convs):
            convs.append(
                nn.Conv1d(
                    input_size,
                    output_size if i == self.n_convs - 1 else input_size,
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

        self._maybe_init(x.shape[2])

        if isinstance(self.residual, bool):
            x_orig = x
            use_residual = True
            use_res_conv = True

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        x = self.dropout(x)

        out = self.encoder(x)

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


class TransformerTimeSeriesModel(TorchTrainableTimeSeriesModel):
    def __init__(self, n_outputs=None, n_heads=1, n_feats=2048, n_convs=1, positional_encoding=None, pos_feats=None, dropout=0.0, residual=False, embedding_dimensions=None, normalization=None, fuse_method="concat", **kwargs):
        super().__init__(**kwargs)

        self.model = TransformerModel(
            n_outputs=n_outputs,
            n_heads=n_heads,
            n_feats=n_feats,
            n_convs=n_convs,
            positional_encoding=positional_encoding,
            pos_feats=pos_feats,
            dropout=dropout,
            residual=residual,
            embedding_dimensions=embedding_dimensions,
            normalization=normalization,
            fuse_method=fuse_method,
        )
