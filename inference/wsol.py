import os

import torch

from utils.wsol import output_cam_segmentations
from .inference import InferenceMethod


class WSOLInference(InferenceMethod):
    """Class for performing Weakly Supervised Object Localization (WSOL) inference using the Class Activation Mapping (CAM) method.
    This class is designed to work with a model and a dataset, and it saves the output CAM segmentations to a specified folder.
    """
    def __init__(self, thresholds, datasets, save_folder, enriched_wsol_output, model_kwargs, **kwargs):
        """
        Args:
            thresholds (list or tuple): List or tuple of thresholds for CAM segmentation.
            datasets (list): List of datasets to perform inference on.
            save_folder (str): Folder where the output segmentations will be saved.
            enriched_wsol_output (bool): Flag indicating whether to use enriched WSOL output (with test time augmentation).
            model_kwargs (dict): Additional keyword arguments for the model.
            **kwargs: Additional keyword arguments for the inference method.
        """
        self.thresholds = thresholds if isinstance(thresholds, (tuple, list)) else [thresholds]
        self.datasets = datasets
        self.save_folder = save_folder
        self.enriched_wsol_output = enriched_wsol_output
        self.model_kwargs = model_kwargs

        self.kwargs = kwargs

    @torch.no_grad()
    def __call__(self, model, device):
        append_thresh = len(self.thresholds) > 1

        for thresh in self.thresholds:
            for dataset in self.datasets:
                if append_thresh:
                    target_folder = os.path.join(self.save_folder + "_" + str(thresh))
                else:
                    target_folder = self.save_folder

                output_cam_segmentations(
                    model=model,
                    dataset=dataset,
                    target_folder=target_folder,
                    device=device,
                    cam_threshold=thresh,
                    enriched_output=self.enriched_wsol_output,
                    **self.model_kwargs,
                    **self.kwargs,
                )