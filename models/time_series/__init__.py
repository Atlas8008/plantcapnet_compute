import re

from .moving_avg_time_series import *
from .lstm_gru import RNNTimeSeriesModel
from .transformer import TransformerTimeSeriesModel
from .conv import ConvolutionalTimeSeriesModel
from .kalman import KalmanFilterTimeSeriesModel
from .base_time_series_model import BaseTimeSeriesModel, TrainableTimeSeriesModel, TorchTrainableTimeSeriesModel


def get_time_series_model(spec, n_out):
    if spec.startswith("ma"):
        spec = spec[2:]

        param_dict = {}

        while len(spec) > 0:
            if match := re.match(r"n([0-9]+)", spec):
                param_dict["n"] = int(match.group(1))
            elif match := re.match(r"b([0-9]+\.[0-9]+)", spec):
                param_dict["base"] = float(match.group(1))
            elif match := re.match(r"a([a-z]+)", spec):
                param_dict["alignment"] = match.group(1)
            elif match := re.match(r"s([0-9]+)", spec):
                param_dict["sigma"] = int(match.group(1))
            elif match := re.match(r"m(lin|gauss|exp)", spec):
                param_dict["mode"] = match.group(1)
            else:
                raise ValueError("Invalid parameter string: " + spec)

            spec = spec[len(match.group(0)):]

        # n = int(match.group(1))
        # base = float(match.group(2))
        # alignment = match.group(3)

        return MovingAverageTimeSeriesModel(
            **param_dict
            # n=n,
            # base=base,
            # alignment=alignment,
        )

    elif spec.startswith("kal"):
        spec = spec[3:]

        param_dict = {
            "trend_kinds": [],
        }

        while len(spec) > 0:
            if match := re.match(r"fep", spec):
                param_dict["fit_each_prediction"] = True
            elif match := re.match(r"nf", spec):
                param_dict["no_fit"] = True
            elif match := re.match(r"tk(lt|ll|s)", spec):
                param_dict["trend_kinds"].append(match.group(1))
            elif match := re.match(r"pn([0-9-.e]+)", spec):
                param_dict["process_noise"] = float(match.group(1))
            elif match := re.match(r"mn([0-9-.e]+)", spec):
                param_dict["measurement_noise"] = float(match.group(1))
            elif match := re.match(r"vm([0-9-.e]+)", spec):
                param_dict["velocity_multi"] = float(match.group(1))
            elif match := re.match(r"sp([0-9-.e]+)", spec):
                param_dict["season_period"] = float(match.group(1))
            elif match := re.match(r"sk([0-9]+)", spec):
                param_dict["season_K"] = int(match.group(1))
            else:
                raise ValueError("Invalid parameter string: " + spec)

            spec = spec[len(match.group(0)):]

        if len(param_dict["trend_kinds"]) == 0:
            param_dict["trend_kinds"].append("lt")

        return KalmanFilterTimeSeriesModel(n_outputs=n_out, **param_dict)
    elif spec.startswith(("lstm", "bilstm", "gru", "bigru")):
        param_dict = {}

        if spec.startswith("bilstm"):
            spec = spec[len("bilstm"):]
            param_dict["rnn_kind"] = "lstm"
            param_dict["bidirectional"] = True
        elif spec.startswith("lstm"):
            spec = spec[len("lstm"):]
            param_dict["rnn_kind"] = "lstm"
        elif spec.startswith("bigru"):
            spec = spec[len("bigru"):]
            param_dict["rnn_kind"] = "gru"
            param_dict["bidirectional"] = True
        elif spec.startswith("gru"):
            spec = spec[len("gru"):]
            param_dict["rnn_kind"] = "gru"

        while len(spec) > 0:
            if match := re.match(r"b([0-9]+)", spec):
                param_dict["train_batch_size"] = int(match.group(1))
            elif match := re.match(r"e([0-9]+)", spec):
                param_dict["train_epochs"] = int(match.group(1))
            elif match := re.match(r"lr?([0-9]+\.[0-9]+)", spec):
                param_dict["train_lr"] = float(match.group(1))
            elif match := re.match(r"lo(mae|r?mse)", spec):
                param_dict["train_loss"] = match.group(1)
            elif match := re.match(r"w([0-9]+\.[0-9]+)", spec):
                param_dict["weight_decay"] = float(match.group(1))
            elif match := re.match(r"p([0-9]+)", spec):
                param_dict["train_pad"] = int(match.group(1))
            elif match := re.match(r"s([0-9]+)", spec):
                param_dict["train_slice"] = int(match.group(1))
            elif match := re.match(r"n([0-9]+)", spec):
                param_dict["num_layers"] = int(match.group(1))
            elif match := re.match(r"h([0-9]+)", spec):
                param_dict["hidden_units"] = int(match.group(1))
            elif match := re.match(r"c([0-9]+)", spec):
                param_dict["n_convs"] = int(match.group(1))
            elif match := re.match(r"res([0-9]+)?", spec):
                if match.group(1) and len(match.group(1)) > 0:
                    param_dict["residual"] = int(match.group(1))
                param_dict["residual"] = True
            elif match := re.match(r"ffe([0-9]+)", spec):
                param_dict["embedding_dimensions"] = int(match.group(1))
            elif match := re.match(r"ffn(l|i|none)", spec):
                param_dict["normalization"] = match.group(1)
            elif match := re.match(r"ffm(add|concat)", spec):
                param_dict["fuse_method"] = match.group(1)
            else:
                raise ValueError("Invalid parameter string: " + spec)

            spec = spec[len(match.group(0)):]

        print("RNN parameters:")
        print(param_dict)

        return RNNTimeSeriesModel(
            n_outputs=n_out,
            **param_dict
        )
    elif spec.startswith(("tf")):
        param_dict = {}

        spec = spec[len("tf"):]

        while len(spec) > 0:
            if match := re.match(r"b([0-9]+)", spec):
                param_dict["train_batch_size"] = int(match.group(1))
            elif match := re.match(r"e([0-9]+)", spec):
                param_dict["train_epochs"] = int(match.group(1))
            elif match := re.match(r"lr?([0-9]+\.[0-9]+)", spec):
                param_dict["train_lr"] = float(match.group(1))
            elif match := re.match(r"lo(mae|r?mse)", spec):
                param_dict["train_loss"] = match.group(1)
            elif match := re.match(r"w([0-9]+\.[0-9]+)", spec):
                param_dict["weight_decay"] = float(match.group(1))
            elif match := re.match(r"p([0-9]+)", spec):
                param_dict["train_pad"] = int(match.group(1))
            elif match := re.match(r"s([0-9]+)", spec):
                param_dict["train_slice"] = int(match.group(1))
            elif match := re.match(r"h([0-9]+)", spec):
                param_dict["n_heads"] = int(match.group(1))
            elif match := re.match(r"f([0-9]+)", spec):
                param_dict["n_feats"] = int(match.group(1))
            elif match := re.match(r"c([0-9]+)", spec):
                param_dict["n_convs"] = int(match.group(1))
            elif match := re.match(r"pe((rel|abs)(sc)?)", spec):
                param_dict["positional_encoding"] = match.group(1)
            elif match := re.match(r"pf([0-9]+)", spec):
                param_dict["pos_feats"] = int(match.group(1))
            elif match := re.match(r"d([0-9]+\.[0-9]+)", spec):
                param_dict["dropout"] = float(match.group(1))
            elif match := re.match(r"res([0-9]+)?", spec):
                if match.group(1) and len(match.group(1)) > 0:
                    param_dict["residual"] = int(match.group(1))
                param_dict["residual"] = True
            elif match := re.match(r"ffe([0-9]+)", spec):
                param_dict["embedding_dimensions"] = int(match.group(1))
            elif match := re.match(r"ffn(l|i|none)", spec):
                param_dict["normalization"] = match.group(1)
            elif match := re.match(r"ffm(add|concat)", spec):
                param_dict["fuse_method"] = match.group(1)
            else:
                raise ValueError("Invalid parameter string: " + spec)

            spec = spec[len(match.group(0)):]

        print("Transformer parameters:")
        print(param_dict)

        return TransformerTimeSeriesModel(
            n_outputs=n_out,
            **param_dict
        )
    elif spec.startswith(("conv")):
        param_dict = {}

        spec = spec[len("conv"):]

        while len(spec) > 0:
            if match := re.match(r"b([0-9]+)", spec):
                param_dict["train_batch_size"] = int(match.group(1))
            elif match := re.match(r"e([0-9]+)", spec):
                param_dict["train_epochs"] = int(match.group(1))
            elif match := re.match(r"lr?([0-9]+\.[0-9]+)", spec):
                param_dict["train_lr"] = float(match.group(1))
            elif match := re.match(r"lo(mae|r?mse)", spec):
                param_dict["train_loss"] = match.group(1)
            elif match := re.match(r"w([0-9]+\.[0-9]+)", spec):
                param_dict["weight_decay"] = float(match.group(1))
            elif match := re.match(r"p([0-9]+)", spec):
                param_dict["train_pad"] = int(match.group(1))
            elif match := re.match(r"s([0-9]+)", spec):
                param_dict["train_slice"] = int(match.group(1))
            elif match := re.match(r"k([0-9]+)", spec):
                param_dict["kernel_size"] = int(match.group(1))
            elif match := re.match(r"f([0-9]+)", spec):
                param_dict["n_feats"] = int(match.group(1))
            elif match := re.match(r"c([0-9]+)", spec):
                param_dict["n_convs"] = int(match.group(1))
            elif match := re.match(r"res([0-9]+)?", spec):
                if match.group(1) and len(match.group(1)) > 0:
                    param_dict["residual"] = int(match.group(1))
                param_dict["residual"] = True
            elif match := re.match(r"ffe([0-9]+)", spec):
                param_dict["embedding_dimensions"] = int(match.group(1))
            elif match := re.match(r"ffn(l|i|none)", spec):
                param_dict["normalization"] = match.group(1)
            elif match := re.match(r"ffm(add|concat)", spec):
                param_dict["fuse_method"] = match.group(1)
            elif match := re.match(r"dwt", spec):
                param_dict["depthwise"] = True
            else:
                raise ValueError("Invalid parameter string: " + spec)

            spec = spec[len(match.group(0)):]

        print("Convolutional parameters:")
        print(param_dict)

        return ConvolutionalTimeSeriesModel(
            n_outputs=n_out,
            **param_dict
        )
    else:
        raise ValueError("Invalid time series model spec: " + spec)

