from turtle import forward
import torch

from torch import nn
from copy import deepcopy


class FocusWeighting(nn.Module):
    def __init__(self, in_channels, inter_channels) -> None:
        super().__init__()

        self.global_informer = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, (1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inter_channels, inter_channels, (1, 1)),
            nn.ReLU(),
        )
        self.local_informer = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, (1, 1)),
            nn.ReLU(),
        )
        self.fuser = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(inter_channels, 1, (1, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        global_infos = self.global_informer(x)
        local_infos = self.local_informer(x)

        infos = global_infos + local_infos

        weighting_residuals = self.fuser(infos)

        weightings = weighting_residuals + 1

        # Normalize
        weightings = weightings / torch.mean(weightings, dim=(2, 3), keepdim=True)

        return weightings


class CoverPredictionConvsWSegmentation(nn.Module):
    def __init__(self, in_feats, nr_outputs):
        super().__init__()

        self.pre_initial_conv = nn.LazyConv2d(in_feats, (1, 1))
        self.segmentation_conv = None

        self.initial_conv = nn.Conv2d(in_feats, in_feats, (1, 1))

        #self.threshold = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        self.class_conv = nn.Conv2d(in_feats, nr_outputs, (1, 1))
        self.bg_irr_conv = nn.Conv2d(in_feats, 2, (1, 1))

        #nn.init.xavier_uniform_(self.threshold)
        nn.init.xavier_uniform_(self.class_conv.weight)
        nn.init.xavier_uniform_(self.bg_irr_conv.weight)
        #nn.init.xavier_uniform_(self.threshold.weight)

        nn.init.constant_(self.class_conv.bias, 0)
        nn.init.constant_(self.bg_irr_conv.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            pre_segmentations = self.segmentation_conv(x, mode="segmentation")

        x = torch.concat([x, pre_segmentations], dim=1)

        x = self.pre_initial_conv(x)
        x = torch.relu(x)
        x = self.initial_conv(x)
        x = torch.relu(x)

        class_scores = self.class_conv(x)
        bg_irr_scores = self.bg_irr_conv(x)

        #scores_full = torch.cat([class_scores, bg_irr_scores], dim=1)

        return class_scores, bg_irr_scores

    def set_model(self, model):
        self.segmentation_conv = deepcopy(model.segmentation_module.segmentation_module)


class CoverPredictionConvs(nn.Module):
    def __init__(self, in_feats, nr_outputs, with_bg_irr=True, apply_gap=False):
        super().__init__()

        self.initial_conv = nn.Conv2d(in_feats, in_feats, (1, 1))

        #self.threshold = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        self.class_conv = nn.Conv2d(in_feats, nr_outputs, (1, 1))
        nn.init.xavier_uniform_(self.class_conv.weight)
        nn.init.constant_(self.class_conv.bias, 0)

        if with_bg_irr:
            self.bg_irr_conv = nn.Conv2d(in_feats, 2, (1, 1))
            nn.init.xavier_uniform_(self.bg_irr_conv.weight)
            nn.init.constant_(self.bg_irr_conv.bias, 0)
        else:
            self.bg_irr_conv = None

        if apply_gap:
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pooling = nn.Identity()

    def forward(self, x):
        if hasattr(self, "pooling"):
            x = self.pooling(x)

        x = self.initial_conv(x)
        x = torch.relu(x)

        class_scores = self.class_conv(x)

        if self.bg_irr_conv is not None:
            bg_irr_scores = self.bg_irr_conv(x)
            return class_scores, bg_irr_scores

        return class_scores,

        #scores_full = torch.cat([class_scores, bg_irr_scores], dim=1)



class SigmoidalCoverPredictionHead(nn.Module):
    def __init__(self, norm_type="l1"):
        super().__init__()

        # self.initial_conv = nn.Conv2d(in_feats, in_feats, (1, 1))

        # #self.threshold = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        # self.class_conv = nn.Conv2d(in_feats, nr_outputs, (1, 1))
        # self.bg_irr_conv = nn.Conv2d(in_feats, 2, (1, 1))

        self.norm_type = norm_type

        self.thresh_one = torch.ones((1, 1, 1, 1))
        self.threshold = nn.Linear(1, 1, bias=False)

        #nn.init.xavier_uniform_(self.threshold)
        # nn.init.xavier_uniform_(self.class_conv.weight)
        # nn.init.xavier_uniform_(self.bg_irr_conv.weight)
        nn.init.xavier_uniform_(self.threshold.weight)

        # nn.init.constant_(self.class_conv.bias, 0)
        # nn.init.constant_(self.bg_irr_conv.bias, 0)

    def forward(self, class_feats, bg_irr_feats=None, weightings=None, mode="cover_prediction"):
        # x = self.initial_conv(x)
        # x = torch.relu(x)

        # class_feats = self.class_conv(x)
        # bg_irr_feats = self.bg_irr_conv(x)

        #class_feats = x[:, :-2]
        #bg_irr_feats = x[:, -2:]

        class_feats = class_feats.to(torch.float32)
        bg_irr_feats = bg_irr_feats.to(torch.float32)

        corr_bg_irr_vals, non_plant_prob = self._apply_threshold(
            class_feats,
            bg_irr_feats,
            norm_type=self.norm_type,
        )

        bg_val = corr_bg_irr_vals[:, :1]
        irr_val = corr_bg_irr_vals[:, 1:]

        bg_val = bg_val.to(torch.float32)
        irr_val = irr_val.to(torch.float32)

        classifications = torch.sigmoid(class_feats)

        if mode is None:
            return classifications, bg_val, irr_val

        if weightings is not None:
            return self.final_prediction_multimode(
                weightings * classifications,
                weightings * bg_val,
                weightings * irr_val,
                mode=mode,
            )
        else:
            return self.final_prediction_multimode(
                classifications, bg_val, irr_val, mode=mode
            )

    def final_prediction_multimode(self, *args, mode, **kwargs):
        was_multimode = True
        outputs = {}

        if not isinstance(mode, (tuple, list)):
            mode = [mode]
            was_multimode = False

        for single_mode in mode:
            output = self.final_prediction(*args, mode=single_mode, **kwargs)

            outputs[single_mode] = output

        if not was_multimode:
            assert len(outputs) == 1
            return list(outputs.values())[0]
        else:
            return outputs

    def final_prediction(self, classifications, bg_val, irr_val, mode):
        if mode.startswith("cover_prediction_mc") and not self.training or mode.startswith("cover_prediction") and not mode.startswith("cover_prediction_mc"):
            # Values are the sum of all plant pixels divided by the number of relevant pixels
            cover_values = torch.sum(classifications, dim=(2, 3)) / (torch.sum(1 - irr_val, dim=(2, 3)))

            return cover_values
        elif mode.startswith("cover_prediction_mc"):
            n_elems = classifications.shape[2] * classifications.shape[3]
            n_samples = int(n_elems * 0.05)
            # Values are the sum of all plant pixels divided by the number of relevant pixels
            coords_dim2 = torch.randint(0, classifications.shape[2], (n_samples,))
            coords_dim3 = torch.randint(0, classifications.shape[3], (n_samples,))

            classifications = classifications[:, :, coords_dim2, coords_dim3]
            irr_val = irr_val[:, :, coords_dim2, coords_dim3]

            cover_values = torch.sum(classifications, dim=2) / (torch.sum(1 - irr_val, dim=2))

            return cover_values
        elif mode == "segmentation":
            return torch.cat([classifications, bg_val, irr_val], dim=1)
        else:
            raise ValueError("Invalid mode: " + mode)

    def _apply_threshold(self, class_feats, bg_irr_feats, norm_type="l1", stop_gradient=False):
        if class_feats.get_device() != -1:
            self.thresh_one = self.thresh_one.to(class_feats.get_device())

        # Norm threshold
        threshold = self.threshold(self.thresh_one)

        # Apply softmax and multiplication
        if norm_type == "l1":
            threshold = torch.sigmoid(threshold)
            _sum = torch.sum(torch.sigmoid(class_feats), dim=1, keepdim=True) + threshold
            non_plant_prob = threshold / _sum
        elif norm_type == "softmax":
            threshold = torch.exp(threshold)
            _sum = torch.sum(torch.exp(class_feats), dim=1, keepdim=True) + threshold
            non_plant_prob = threshold / _sum
        else:
            raise ValueError()

        bg_irrelevant_decider_values = torch.softmax(bg_irr_feats, dim=1)

        if stop_gradient:
            non_plant_prob = non_plant_prob.detach()

        corrected_bg_irrelevant_decider_values = non_plant_prob * bg_irrelevant_decider_values

        return corrected_bg_irrelevant_decider_values, non_plant_prob


class SimpleCoverPredictionHead(nn.Module):
    def __init__(self, act_fun="sigmoid", bg_class=False, clip=False):
        super().__init__()

        if act_fun == "sigmoid":
            self.activation = nn.Sigmoid()
        elif act_fun == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif act_fun == "linear":
            self.activation = nn.Identity()

        self.clip = clip
        self.bg_class = bg_class

    def forward(self, class_feats, weightings=None, mode="cover_prediction"):
        class_feats = class_feats.to(torch.float32)

        classifications = self.activation(class_feats)

        if mode is None:
            return classifications,

        if weightings is not None:
            classifications = weightings * classifications

        return self.final_prediction_multimode(
            classifications, mode=mode,
        )

    def final_prediction_multimode(self, *args, mode, **kwargs):
        was_multimode = True
        outputs = {}

        if not isinstance(mode, (tuple, list)):
            mode = [mode]
            was_multimode = False

        for single_mode in mode:
            output = self.final_prediction(*args, mode=single_mode, **kwargs)

            outputs[single_mode] = output

        if not was_multimode:
            assert len(outputs) == 1
            return list(outputs.values())[0]
        else:
            return outputs

    def final_prediction(self, classifications, mode):
        if mode == "cover_prediction":
            if not self.bg_class:
                cover_values = torch.mean(classifications, dim=(2, 3))
            else:
                cover_values = torch.mean(classifications[:, :-1], dim=(2, 3))

            if hasattr(self, "clip") and self.clip:
                cover_values = torch.clamp(cover_values, 1e-4, 1.0)

            return cover_values
        elif mode == "segmentation":
            return classifications
        else:
            raise ValueError("Invalid mode: " + mode)