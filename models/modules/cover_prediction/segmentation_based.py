import os
import torch
import torch.nn.functional as F

from torch import nn

from utils.postprocessing import predict_enriched_output_by_augmentation

from .modules import zeroshot_phenology_prediction
from .utils import discretization_utils, background_utils, tiling_utils, relevance_utils
from .asymm_segmentation_mappers import MAPPERS



__all__ = [
    "get_segmentation_based_cover_prediction_module",
    "SegmentationBasedCoverPredictionModule",
]

def get_segmentation_based_cover_prediction_module(**kwargs):
    return SegmentationBasedCoverPredictionModule(
        **kwargs,
    )


def interpolate_all(x, *args, **kwargs):
    out = []

    if not isinstance(x, (tuple, list)):
        x = [x]

    for item in x:
        out.append(F.interpolate(item, *args, **kwargs))

    if len(out) == 1:
        out = out[0]
    else:
        out = tuple(out)

    return out


class SegmentationBasedCoverPredictionModule(nn.Module):
    def __init__(self, relevance_model, relevance_model_preprocessing=None, preprocessing=None, flow_pheno_model=None, sen_pheno_model=None, class_mapper=None, include_dead_litter=False):
        """
        Segmentation-based (zero-shot) cover prediction module.

        Args:
            relevance_model (str): Relevance model to use. Can be "mindelta", "none" or a path to a file. The model classifying the input data into relevant and irrelevant areas for cover calculation.
            relevance_model_preprocessing (str): Preprocessing for the relevance model.
            preprocessing (str): Preprocessing for the input data.
            flow_pheno_model (str): Flowering phenology model to use. Should be a name of an existing model defined in the zeroshot_phenology_prediction module.
            sen_pheno_model (str): Senescence phenology model to use. Should be a name of an existing model defined in the zeroshot_phenology_prediction module.
            class_mapper (str): Class mapper to use. Can be None, "none" or the name of a predefined mapper.
            include_dead_litter (bool): If True, include dead litter in the output. Will be added as a separate channel comprised of zeros.
        """
        super().__init__()

        self.model = None
        if class_mapper not in (None, "none"):
            self.mapper = MAPPERS[class_mapper]()
        else:
            self.mapper = None

        self.include_dead_litter = include_dead_litter
        if os.path.isfile(relevance_model):
            self.relevance_model = torch.load(relevance_model, map_location=torch.device("cpu"), weights_only=False)
        elif relevance_model.lower() == "mindelta":
            self.relevance_model = relevance_utils.MinimumDeltaIrrelevanceModule(delta=0.01)
        elif relevance_model.lower() == "none":
            self.relevance_model = None
        else:
            raise ValueError("Invalid relevance model: " + str(relevance_model))

        self.relevance_model_preprocessing = relevance_model_preprocessing
        self.preprocessing = preprocessing

        self.flow_pheno_model = None
        self.sen_pheno_model = None

        if flow_pheno_model is not None:
            self.flow_pheno_model = zeroshot_phenology_prediction.get_phenology_module(flow_pheno_model)
        if sen_pheno_model is not None:
            self.sen_pheno_model = zeroshot_phenology_prediction.get_phenology_module(sen_pheno_model)

    def forward(
            self,
            x,
            mode,
            submodel_kwargs,
            enriched_prediction=False,
            interlaced_prediction=True,
            enrichment_params="s1,s0.5,v,h",
            interlace_size=512,
            discretization="combined",
            bg_type="zero",
        ):
        """
        Forward pass of the segmentation based cover prediction module.
        Args:
            x (torch.Tensor): Input tensor.
            mode (str): Mode of the module. Can be "cover_prediction" or "segmentation".
            submodel_kwargs (dict): Additional arguments for the submodel.
            enriched_prediction (bool): If True, use enriched prediction.
            interlaced_prediction (bool): If True, use interlaced prediction.
            enrichment_params (str): Parameters for test time augmentation. Can be a comma-separated list of values.
                - "s<float>" - Scale factor. Can be a float or a comma-separated list of floats. E.g. "s1,s0.5".
                - "v" - Vertical flip.
                - "h" - Horizontal flip.
            interlace_size (int): Size of the interlace tiles. Recommended to be 224, 256, 448 or 512.
            discretization (str): Discretization method. One of "argmax", "scaled", "scaled_thresh", "none".
            bg_type (str): Background type.
        Returns:
            torch.Tensor: Output tensor.
        """
        assert self.model is not None, "Module.model has to be set before calling forward"

        if interlace_size <= 0:
            interlaced_prediction = False

        if not self.model[0].training and enriched_prediction:
            scales = []
            vflip = False
            hflip = False

            for param in enrichment_params.split(","):
                if param == "v":
                    vflip = True
                elif param == "h":
                    hflip = True
                elif param.startswith("s"):
                    scales.append(float(param[1:]))
                else:
                    raise ValueError("Invalid enrichment param: " + param)

            prediction_function = \
                lambda x: interpolate_all(
                    predict_enriched_output_by_augmentation(
                    x,
                    self.model[0],
                    device=x.get_device(),
                    scales=tuple(scales),
                    vertical_flips=vflip,
                    horizontal_flips=hflip,
                    interlaced=interlaced_prediction,
                    interlace_tile_size=(interlace_size,),
                    model_mode=None,
                    model_kwargs=submodel_kwargs,
                ),
                (x.shape[2], x.shape[3]),
                mode="bilinear"
            )
        else:
            prediction_function = \
                lambda x: self.model[0](x, **submodel_kwargs)

        cam = prediction_function(x)

        # Map long class list to short one
        if self.mapper is not None:
            cam = self.mapper(cam)

        # Check, if prediction consists of two outputs
        if isinstance(cam, (tuple, list)):
            cam, deoc_cam = cam

            deoc_segmentations = torch.sigmoid(deoc_cam)
        else:
            deoc_cam = None
            deoc_segmentations = None

        segmentations = torch.sigmoid(cam)

        segmentations = self.maybe_add_dead_litter(segmentations)
        segmentations, bg = background_utils.process_background(
            segmentations,
            input_image=x,
            bg_type=bg_type
        )

        segmentations = discretization_utils.apply_discretization(
            segmentations,
            deoc_segmentations,
            d_type=discretization,
            input_img=x,
            prediction_fun=prediction_function,
        )


        if self.relevance_model is not None:
            self.relevance_model = self.relevance_model.to(x.get_device())

        with torch.no_grad():
            relevance = relevance_utils.get_relevance(
                x,
                self.relevance_model,
                source_preprocessing=self.preprocessing,
                target_preprocessing=self.relevance_model_preprocessing,
            )

        # Apply relevance
        segmentations *= relevance

        pheno_map = self.predict_phenology(x, segmentations)

        if isinstance(mode, tuple):
            modes = mode
            multimode = True
        else:
            modes = mode,
            multimode = False

        output_dict = {}

        for mode in modes:
            if mode == "cover_prediction":
                cover_values = torch.sum(segmentations[:, :-1], dim=(2, 3)) / torch.sum(relevance)

                cover_values = torch.nan_to_num(cover_values, nan=0.0, posinf=0.0, neginf=0.0)

                if pheno_map is None:
                    output = cover_values
                else:
                    pheno_vals = torch.sum(pheno_map[:, :-1], dim=(3, 4)) / torch.sum(segmentations[:, :-1], dim=(2, 3))[..., None]

                    output = cover_values, pheno_vals
            elif mode == "segmentation":
                irrelevance = 1 - relevance

                output = torch.cat([segmentations, irrelevance], dim=1)
            else:
                raise ValueError("Invalid mode: " + mode)

            output_dict[mode] = output

        if multimode:
            return output_dict
        else:
            assert len(output_dict) == 1
            return list(output_dict.values())[0]

    def predict_phenology(self, x, segmentations):
        pheno_map = None
        flow_map = None
        sen_map = None

        if self.flow_pheno_model is not None:
            with torch.no_grad():
                flow_map = self.flow_pheno_model(x)

        if self.sen_pheno_model is not None:
            with torch.no_grad():
                sen_map = self.sen_pheno_model(x)

        if flow_map is None and sen_map is None:
            pheno_map = None
        else:
            if flow_map is None and sen_map is not None:
                flow_map = torch.zeros_like(sen_map)
            elif flow_map is not None and sen_map is None:
                sen_map = torch.zeros_like(flow_map)

            pheno_map = torch.stack([flow_map, sen_map], dim=0)
            pheno_map = torch.moveaxis(pheno_map, 0, 2)

            pheno_map = segmentations[:, :, None] * pheno_map

        return pheno_map

    def maybe_add_dead_litter(self, segmentations):
        dead_litter = torch.zeros_like(segmentations[:, 0:1])

        if self.include_dead_litter:
            segmentations = torch.cat([segmentations, dead_litter], dim=1)

        return segmentations