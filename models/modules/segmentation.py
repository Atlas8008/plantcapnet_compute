import torch

from torch import nn

from .deocclusion import SegmentationDeocclusionSwitch, get_deocclusion_module_from_spec


def get_segmentation_module(
        in_feats,
        out_feats,
        segmentation_module_args,
        deoc_module_args
    ):
    segmentation_module_spec = segmentation_module_args["segm_module_spec"]
    deoc_module_spec = deoc_module_args["deoc_module_spec"]

    segmentation_module = get_segmentation_module_from_spec(
        spec=segmentation_module_spec,
        in_feats=in_feats,
        out_feats=out_feats,
    )
    deocclusion_module = get_deocclusion_module_from_spec(
        in_feats=in_feats,
        out_feats=out_feats,
        spec=deoc_module_spec,
    )

    switch = SegmentationDeocclusionSwitch(
        segmentation_module=segmentation_module,
        deocclusion_module=deocclusion_module,
        **segmentation_module_args,
        **deoc_module_args,
    )

    return switch


def get_segmentation_module_from_spec(spec, **kwargs):
    if spec in (None, "none"):
        return DeocclusionSegmentationModule(**kwargs)
    else:
        raise ValueError("Invalid segmentation module spec: " + spec)


class DeocclusionSegmentationModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.class_conv = nn.Conv2d(
            in_feats,
            out_feats,
            kernel_size=(1, 1),
        )

    def forward(self, x, mode="segmentation", output_size=None):
        x = self.class_conv(x)

        if output_size is not None:
            x = nn.functional.interpolate(
                x,
                output_size,
                mode="bilinear"
            )

        x = x.to(torch.float32)

        if mode == "cam":
            return x
        elif mode == "segmentation":
            return torch.sigmoid(x)
