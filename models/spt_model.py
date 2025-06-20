import gc
import math
import torch
import random

from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from contextlib import nullcontext

from utils.torch_utils import indexgrid


class PlantAnalysisModel(nn.Module):
    def __init__(
        self,
        feature_extractor,
        *,
        classification_module=None,
        segmentation_module=None,
        zeroshot_cover_module=None,
        cover_module=None,
        temporal_cover_fusion_module=None,
        predictors=None
        ):
        super().__init__()

        self.feature_extractor = feature_extractor

        self.classification_module = classification_module
        self.segmentation_module = segmentation_module
        self.zeroshot_cover_module = zeroshot_cover_module
        self.cover_module = cover_module
        self.temporal_cover_module = None
        self.temporal_cover_fusion_module = temporal_cover_fusion_module

        self.additional_segmentation_module = None

        self.trainable_time_steps = False
        self.temporal_dim = -1
        self.tmp_share_weights = True

        self.head_modules = {
            "classification": self.classification_module,
            "segmentation": self.segmentation_module,
            "zeroshot_cover_prediction": self.zeroshot_cover_module,
            "cover_prediction": self.cover_module,
            "temporal_cover_fusion_module": self.temporal_cover_fusion_module,
            "temporal_cover_prediction": None,
            "features": None,
            "mean_features": None,
        }

        if predictors is None:
            predictors = {}
        self._predictors = predictors

    @property
    def predictors(self):
        return self._predictors

    @predictors.setter
    def predictors(self, v):
        self._predictors = v

    @torch.no_grad()
    def added_segmodel_get_outputs(self, img, **kwargs):
        self.additional_segmentation_module.eval()
        x_seg = self.additional_segmentation_module.feature_extractor(img)
        out_seg = self.additional_segmentation_module.segmentation_module(x_seg, **kwargs)

        return x_seg, out_seg

    def maybe_get_seg_model_output_params(self, img, **kwargs):
        if self.additional_segmentation_module is not None:
            x_seg, out_seg = self.added_segmodel_get_outputs(img, **kwargs["seg_model_kwargs"]) # TODO

            kwargs = {
                **kwargs,
                **{
                    "feat_proposals": x_seg,
                    "class_proposals": out_seg,
                }
            }
            del kwargs["seg_model_kwargs"]

        return kwargs

    def forward(self, x, head, mc_params=None, **kwargs):
        sensor_data = None
        suppl_data = None

        if isinstance(x, dict):
            img = x["x"]
            if "suppl_data" in x:
                suppl_data = x["suppl_data"]
            if "sensor_data" in x:
                sensor_data = x["sensor_data"]

            x = img
        elif isinstance(x, (tuple, list)):
            img, suppl_data = x
            x = img
        else:
            img = x

        # Initialize the temporal cover prediction head with normal cover prediction weights at first use
        if head == "temporal_cover_prediction" and self.head_modules["temporal_cover_prediction"] is None:
            print("INITIALIZE")
            if self.tmp_share_weights:
                self.head_modules["temporal_cover_prediction"] = self.head_modules["cover_prediction"]
            else:
                self.head_modules["temporal_cover_prediction"] = deepcopy(self.head_modules["cover_prediction"])

        if head in self.predictors:
            predictor = self.predictors[head]
        elif head in self.head_modules:
            predictor = self.head_modules[head]
        else:
            predictor = None

        if head == "classification":
            x = self.feature_extractor(x)
            return predictor(x, **kwargs)
        elif head == "segmentation":
            x = self.feature_extractor(x)
            return predictor(x, output_size=img.shape[2:4], **kwargs)
        elif head == "features":
            x = self.feature_extractor(x)
            return x
        elif head == "mean_features":
            x = self.feature_extractor(x)
            return torch.mean(x, dim=(2, 3))
        elif head == "zeroshot_cover_prediction":
            return predictor(x, **kwargs)
        elif head in ("cover_prediction", "temporal_cover_prediction"):
            self.feature_extractor.eval() # TODO: Make this more flexible

            if mc_params in (None, "none") or not self.training:
                kwargs = self.maybe_get_seg_model_output_params(img, **kwargs)
                x = self.feature_extractor(x)

                x = self._maybe_fuse_suppl_data(
                    x,
                    suppl_data,
                    mc_params=mc_params,
                    n_dim=self.temporal_dim,
                    **kwargs,
                )

                return predictor(x, img=img, sensor_data=sensor_data, **kwargs)
            else:
                return self.mc_prediction(
                    mc_params=mc_params,
                    x=x,
                    feature_extractor=self.feature_extractor,
                    predictor=predictor,
                    suppl_data=suppl_data,
                    sensor_data=sensor_data,
                    **kwargs,
                )
        else:
            raise ValueError("Invalid head: " + head)

    def mc_prediction(self, mc_params, x, feature_extractor, predictor, suppl_data=None, sensor_data=None, **kwargs): # TODO: Discount multicounts
        params = mc_params.split(",")

        if len(params) == 3:
            type_ = params[0]
            img_size, bs = map(int, params[1:])
        else:
            type_ = "default"
            img_size, bs = map(int, params)

        orig_bs, fs, _, _ = x.shape

        samples = self._get_mc_samples(
            x,
            type_,
            img_size,
            bs,
        )

        stacked = torch.stack(samples, dim=0)
        batched = stacked.reshape((-1, fs, img_size, img_size))

        kwargs = self.maybe_get_seg_model_output_params(batched, **kwargs)
        x = feature_extractor(batched)

        if suppl_data is not None:
            suppl_data = suppl_data.repeat(
                x.shape[0], *([1] * (len(suppl_data.shape) - 1))
            )

            #kwargs["suppl_data"] = suppl_data

            x = self._maybe_fuse_suppl_data(
                x,
                suppl_data,
                mc_params=mc_params,
                **kwargs,
            )

        if sensor_data is not None:
            sensor_data = sensor_data.repeat(
                x.shape[0], *([1] * (len(sensor_data.shape) - 1))
            )

        predicted = predictor(x, img=batched, sensor_data=sensor_data, **kwargs)

        # Aggregate
        if not isinstance(predicted, (tuple, list)):
            predicted = [predicted]

        predicted_aggregated = []

        for p in predicted:
            single_shape = p.shape[1:]
            p = p.reshape((orig_bs, bs, *single_shape))
            p = torch.mean(p, dim=1)

            predicted_aggregated.append(p)

            #predicted = predicted.reshape((orig_bs, bs, -1))
            #predicted = torch.mean(predicted, dim=1)

        if len(predicted_aggregated) == 1:
            predicted_aggregated = predicted_aggregated[0]
        else:
            predicted_aggregated = tuple(predicted_aggregated)

        return predicted_aggregated

    def _get_mc_samples(self, x, type_, img_size, bs):
        _, f_, h, w = x.shape

        if type_ == "default":
            xs = torch.randint(0, w - img_size, (bs,))
            ys = torch.randint(0, h - img_size, (bs,))
        elif type_ == "padded":
            x = F.pad(
                x,
                (img_size - 1, img_size - 1, img_size - 1, img_size - 1),
                mode="reflect",
            )

            xs = torch.randint(0, w - 1, (bs,))
            ys = torch.randint(0, h - 1, (bs,))
        elif type_ in ("grid", "rgrid"):
            n_cells_y = math.ceil(h / img_size)
            n_cells_x = math.ceil(w / img_size)

            x = nn.functional.interpolate(
                x,
                size=(
                    img_size * n_cells_y,
                    img_size * n_cells_x,
                ),
                mode="bilinear"
            )

            indices = indexgrid((n_cells_y, n_cells_x), x.get_device()).reshape((2, -1))

            selected_indices = torch.randperm(bs)

            ys, xs = indices[:, selected_indices]

            if type_ == "rgrid":
                x_offset = random.randint(-img_size // 2, img_size // 2 - 1)
                y_offset = random.randint(-img_size // 2, img_size // 2 - 1)

                x = F.pad(
                    x,
                    (
                        -min(x_offset, 0),  # padding left
                        max(x_offset, 0),   # padding right
                        -min(y_offset, 0),  # padding top
                        max(y_offset, 0),   # padding bottom
                    ),
                    value=0,
                )
        else:
            raise ValueError("Invalid MC type: " + type_)

        samples = []

        for x_val, y_val in zip(xs, ys):
            sample_batch = x[:, :, y_val:y_val + img_size, x_val:x_val + img_size]

            samples.append(sample_batch)

        return samples

    def _maybe_fuse_suppl_data(self, x, suppl_data, mc_params, dropout=0.0, **kwargs):
        # Fuse temporal data into the prediction
        # Dropout of additional info with a 50% chance during training
        if suppl_data is not None:
            # Check, if supplementary data is an image or a series of images; if yes: predict cover
            if len(suppl_data.shape) >= 4:
                suppl_data = self.predict_suppl_data(
                    suppl_data,
                    head="cover_prediction",
                    mc_params=mc_params,
                    **kwargs,
                )
            else: # Assume that it's already cover data
                pass

            # Fuse it into x
            x = self.temporal_cover_fusion_module(suppl_data, x)

            # Maybe apply dropout
            if self.training and random.uniform(0, 1) < dropout:
                x *= 0

            # Remove suppl_data after fusing it
            suppl_data = None

        return x

    #@torch.no_grad()
    def predict_suppl_data(self, suppl_data_tensor, n_dim=None, **model_kwargs):
        if self.trainable_time_steps:
            context_manager = nullcontext()
        else:
            context_manager = torch.no_grad()

        with context_manager:
            # Multiple inputs
            if len(suppl_data_tensor.shape) == 5:
                assert n_dim is not None

                if n_dim == -1: # We need this to be positive for the dim_indexer function
                    n_dim = len(suppl_data_tensor.shape) - 1

                indexer = (slice(None),) * len(suppl_data_tensor.shape)

                dim_indexer = lambda i: indexer[:n_dim] + (i,) + indexer[n_dim + 1:]

                pre_outs = [
                    self(
                        suppl_data_tensor[dim_indexer(i)],
                        **model_kwargs
                    )
                    for i in range(suppl_data_tensor.shape[n_dim])
                ]

                out = torch.stack(pre_outs, dim=-1)
            else: # Single input
                out = self(
                    suppl_data_tensor,
                    **model_kwargs
                )

            return out


class EnsembleModel(nn.Module):
    def __init__(self, *submodels, dynamic_device=None) -> None:
        super().__init__()

        self.dynamic_device = dynamic_device
        self.submodels = nn.ModuleList(submodels)

    def forward(self, *args, **kwargs):
        # Multi outputs have the shape [(o1_s1, o2_s1, ...), (o1_s2, o2_s2, ...), ...]
        #outputs = [submodel(*args, **kwargs) for submodel in self.submodels]
        # out = self._merge_outputs(outputs)

        outputs = []
        for i, submodel in enumerate(self.submodels):
            if self.dynamic_device is not None:
                self.to("cpu")
                submodel = deepcopy(submodel) # Deepcopy to prevent the need to copy back to cpu and save memory
                submodel.to(self.dynamic_device)
            outputs = self._apply_reduce_outputs(
                outputs + [submodel(*args, **kwargs)],
                action="sum"
            )

            if self.dynamic_device:
                del submodel
                gc.collect()
                torch.cuda.empty_cache()
                #submodel.to("cpu")

            outputs = [outputs]

        out = self._apply_reduce_outputs(
            outputs,
            action="div",
            coef=len(self.submodels)
        )

        return out

    def _apply_reduce_outputs(self, outputs, action="mean", coef=None):
        if isinstance(outputs[0], dict):
            out = {
                k: self._apply_reduce_outputs(
                    [out[k] for out in outputs],
                    action=action,
                    coef=coef,
                ) for k in outputs[0].keys()
            }
        elif isinstance(outputs[0], tuple):
            out = tuple(
                self._apply_reduce_outputs(
                    [out[i] for out in outputs],
                    action=action,
                    coef=coef,
                ) for i in range(len(outputs[0]))
            )
        else:
            if action == "mean":
                out = torch.mean(torch.stack(outputs, dim=0), dim=0)
            elif action == "sum":
                out = torch.sum(torch.stack(outputs, dim=0), dim=0)
            elif action == "div":
                assert coef is not None, "Coefficient has to be provided for div action"
                out = [o / coef for o in outputs]

                if len(outputs) == 1:
                    out = out[0]
            else:
                raise ValueError("Invalid action: " + action)

        return out
