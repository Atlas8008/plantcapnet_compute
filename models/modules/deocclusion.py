import torch

from torch import nn


def get_deocclusion_module_from_spec(spec, in_feats, out_feats):
    """Get a deocclusion module from a specification string. Currently no models are used.

    Args:
        spec (str): The specification string. Currently only "none" is supported.
        in_feats (int): The number of input features.
        out_feats (int): The number of output features.
    Returns:
        nn.Module: The deocclusion module.
    """
    if spec in (None, "none"):
        return None


class SegmentationDeocclusionSwitch(nn.Module):
    """A switch module that combines a segmentation module and a deocclusion module."""
    def __init__(self,
            segmentation_module,
            deocclusion_module,
            training_output_fuse_method="none",
            eval_output_fuse_method="none",
            feature_fuser=None,
            **kwargs,
        ) -> None:
        super().__init__()

        self.segmentation_module = segmentation_module
        self.deocclusion_module = deocclusion_module

        self.training_output_fuse_method = training_output_fuse_method
        self.eval_output_fuse_method = eval_output_fuse_method
        self.feature_fuser = feature_fuser

    def forward(
            self,
            x,
            *,
            use_deocclusion,
            output_size=None,
            segmentation_kwargs=None,
            deocclusion_kwargs=None,
        ):
        segmentation_kwargs = segmentation_kwargs or {}
        deocclusion_kwargs = deocclusion_kwargs or {}

        segmentation_output = self.segmentation_module(x, output_size=output_size, **segmentation_kwargs)

        inputs = []

        if self.feature_fuser is not None:
            inputs = [self.feature_fuser(*inputs)]

        if use_deocclusion:
            deocclusion_output = self.deocclusion_module(
                feats=x,
                segmentation=segmentation_output,
                output_size=output_size,
                **deocclusion_kwargs
            )

            output = self._maybe_combine_class_outputs(
                segmentation_output,
                deocclusion_output
            )
        else:
            output = segmentation_output

        return output

    def _maybe_combine_class_outputs(self, classes_a, classes_b):
        if self.training:
            fuse_method = self.training_output_fuse_method
        else:
            fuse_method = self.eval_output_fuse_method

        if fuse_method == "max":
            return torch.maximum(classes_a, classes_b)
        elif fuse_method == "sum":
            return classes_a + classes_b
        elif fuse_method == "relusum":
            return classes_a + torch.relu(classes_b)
        elif fuse_method == "none":
            if self.training:
                return classes_b
            else:
                return classes_a, classes_b
        else:
            raise ValueError("Invalid fuse_method: " + fuse_method)

