import torch

from torch import nn

from models.modules.cover_prediction.modules.data_injection import get_data_injection_module

from .modules import cover_prediction
from .modules import phenology_prediction

__all__ = ["get_cover_prediction_module", "CoverPredictionModule"]


def get_cover_prediction_module(**kwargs):
    return CoverPredictionModule(**kwargs)


class CoverPredictionModule(nn.Module):
    def __init__(self, in_feats, out_feats, cover_head_type="default", pheno_model=None, dinj_spec=None, **kwargs):
        """
        Module for cover prediction.

        Args:
            in_feats (int): Number of input features.
            out_feats (int): Number of output features.
            cover_head_type (str): Type of head to use for the cover prediction model. Can be 'default' (sigmoidal model with irrelevance prediction), 'simple_sigmoid' (sigmoidal model without irrelevance), 'simple_softmax' (softmax model without background class), 'simple_softmax2' (softmax model with background class) or 'linear' (linear model with global average pooling applied).
            pheno_model (str): Phenology model to use. Can be "fs" (flowering and senescence) or "vfs" (vegetative, flowering, and senescence), None or "none".
            dinj_spec (str): Specification of the data injection module that can be used to, for example, inject sensor information into the network. This is a string that is passed to the data injection module. Refer to the documentation of the data injection module for more information.
        """
        super().__init__()

        kwargs["pheno_model"] = pheno_model


        if cover_head_type == "default":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats)
            self.prediction_head = cover_prediction.SigmoidalCoverPredictionHead(norm_type="l1")
        elif cover_head_type == "defaultsm":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats)
            self.prediction_head = cover_prediction.SigmoidalCoverPredictionHead(norm_type="softmax")
        elif cover_head_type == "segmentation_combined":
            self.convolution_head = cover_prediction.CoverPredictionConvs(
                2 * in_feats, out_feats)
            self.prediction_head = cover_prediction.SigmoidalCoverPredictionHead()
        elif cover_head_type == "simple_sigmoid":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats, with_bg_irr=False)
            self.prediction_head = cover_prediction.SimpleCoverPredictionHead("sigmoid")
        elif cover_head_type == "simple_softmax":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats, with_bg_irr=False)
            self.prediction_head = cover_prediction.SimpleCoverPredictionHead("softmax")
        elif cover_head_type == "simple_softmax2":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats + 1, with_bg_irr=False)
            self.prediction_head = cover_prediction.SimpleCoverPredictionHead("softmax", bg_class=True)
        elif cover_head_type == "linear":
            self.convolution_head = cover_prediction.CoverPredictionConvs(in_feats, out_feats, with_bg_irr=False, apply_gap=True)
            self.prediction_head = cover_prediction.SimpleCoverPredictionHead("linear", clip=True)
        else:
            raise ValueError("Invalid cover_head_type: " + cover_head_type)

        if pheno_model not in (None, "none"):
            self.phenology_module = phenology_prediction.PhenologyModule(
                in_feats=in_feats,
                n_classes=out_feats,
                **kwargs,
            )
        else:
            self.phenology_module = None

        # Data injection module for sensor data and other data
        if dinj_spec not in (None, "none"):
            self.data_injection_module = get_data_injection_module(
                in_feats,
                dinj_spec,
            )
        else:
            self.data_injection_module = None

    def forward(self, x, mode="cover_prediction", feat_proposals=None, class_proposals=None, img=None, sensor_data=None):
        assert feat_proposals is class_proposals is None or \
            feat_proposals is not None and class_proposals is not None and self.class_fuser is not None

        if hasattr(self, "data_injection_module") and self.data_injection_module is not None:
            if sensor_data is not None:
                x = self.data_injection_module(sensor_data, x) # TODO: Make data injection compatible with combined model

        class_outs = self.convolution_head(x)

        class_feats = class_outs

        pred = self.prediction_head(
            *class_outs,
            mode=mode,
        )

        if self.phenology_module is not None:
            pheno_mode_translations = {
                "cover_prediction": "phenology_prediction",
                "cover_prediction_pheno_max": "phenology_max",
                "cover_prediction_pheno_full": ("phenology_prediction", "phenology_max")
            }
            # Selection of multiple modes at once
            if isinstance(mode, tuple):
                pheno_mode = []

                for m in mode:
                    pm = pheno_mode_translations.get(m, m)

                    # Handle the multi-mode case
                    if isinstance(pm, tuple):
                        pheno_mode.extend(pm)
                    else:
                        pheno_mode.append(pm)

                pheno_mode = tuple(pheno_mode)
            else:
                pheno_mode = pheno_mode_translations.get(mode, mode)

            pred_pheno = self.phenology_module(
                classes=torch.sigmoid(class_feats[0]),
                feats=x,
                img=img,
                mode=pheno_mode,
            )
            pred = (pred, pred_pheno)

        return pred

    def set_model(self, model):
        if hasattr(self.convolution_head, "set_model"):
            self.convolution_head.set_model(model)
        if hasattr(self.prediction_head, "set_model"):
            self.prediction_head.set_model(model)