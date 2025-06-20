import os
import torch
import torchvision

from torch import nn

from networks import fpn
from utils.torch_utils import resnet_v1_5_to_v1


class FEWrapper(nn.Module):
    def __init__(self, base_model, modules):
        super().__init__()

        self.base_model = base_model

        self.pipe = nn.Sequential(*modules)

    def forward(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class FPNFEWrapper(nn.Module):
    def __init__(self, fe, modules):
        super().__init__()

        self.base_model = fe.base_model
        self.fpn_layers = fe.fpn_layers

        self.pipe = nn.Sequential(
            fe,
            *modules,
        )

    def forward(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


def get_feature_extractor(base_network, use_fpn, **kwargs):
    if not use_fpn:
        feature_extractor = get_model_backbone(base_network)
        out_feats = 2048
    else:
        feature_extractor, out_feats = get_fpn_feature_extractor(
            base_network,
            **kwargs,
        )

    return feature_extractor, out_feats


def get_fpn_feature_extractor(
        base_network,
        fpn_spec,
        fpn_unetlike,
        fpn_atrous_spec=None,
        fpn_output_agg=None,
        **kwargs
    ):
    spec_components = fpn_spec.split("-")
    fpn_layer = spec_components[:-1]
    fpn_d = spec_components[-1]
    if len(fpn_layer) == 1:
        fpn_layer = fpn_layer[0]

    fpn_d = int(fpn_d)

    out_feats = None

    if fpn_atrous_spec:
        net = fpn.AtrousFPN
        kwargs["atrous_spec"] = fpn_atrous_spec
    else:
        net = fpn.FPN

    if not fpn_unetlike:
        feature_extractor = net(
            base_network,
            out_layer=fpn_layer,
            d=fpn_d,
            use_bn=False,
            upsampling_method="bilinear",
            output_aggregation=fpn_output_agg,
            **kwargs,
        )

        out_feats = fpn_d
    else:
        feature_extractor = net(
            base_network,
            out_layer=fpn_layer,
            d=fpn_d,
            use_bn=False,
            upsampling_method="transposed",
            scaling_feats=True,
            aggregation="concat",
            output_aggregation=fpn_output_agg,
            **kwargs,
        )

        out_feats = fpn_d // 2 ** fpn.FPN.OUTPUT_LAYER_NAMES[::-1].index(fpn_layer)

    if base_network == "resnet50v1":
        resnet_v1_5_to_v1(feature_extractor.base_model)

    return feature_extractor, out_feats


def _resnet_model_to_sequential(model):
    return FEWrapper(
        model,
        [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ],
    )


def get_model_backbone(model_name):
    if os.path.isfile(model_name):
        print("Loaded model from", model_name)
        model = torch.load(model_name, weights_only=False)

        backbone = _resnet_model_to_sequential(model)
    elif model_name in ("resnet50", "resnet50v1", "resnet50v1.5"):
        model = torchvision.models.resnet50(pretrained=True)

        backbone = _resnet_model_to_sequential(model)

        if model_name == "resnet50v1":
            resnet_v1_5_to_v1(backbone)
    elif model_name.lower() in dir(torchvision.models):
        backbone = getattr(torchvision.models, model_name.lower())(pretrained=True)
    else:
        raise ValueError("Unknown model name: " + model_name)

    return backbone