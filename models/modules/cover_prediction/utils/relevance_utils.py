import torch

from torch import nn
from utils.augmentations.preprocessing import Preprocess, DePreprocess


class MinimumDeltaIrrelevanceModule(nn.Module):
    def __init__(self, delta=0.01):
        super().__init__()

        self.delta = delta

    def forward(self, x):
        mean_img = torch.mean(x, dim=1)

        min_v = torch.amin(mean_img, dim=(1, 2), keepdim=False)
        max_v = torch.amax(mean_img, dim=(1, 2), keepdim=False)

        delta_real = self.delta * (max_v - min_v)

        return torch.where(
            mean_img > min_v + delta_real,
            torch.ones_like(mean_img),
            torch.zeros_like(mean_img),
        )


def get_relevance(x, relevance_model, source_preprocessing, target_preprocessing):
    if isinstance(relevance_model, MinimumDeltaIrrelevanceModule):
        return relevance_model(x)
    elif relevance_model is not None:
        if source_preprocessing != target_preprocessing:
            x = Preprocess(
                target_preprocessing,
                preceded_by_scaling=False
            )(DePreprocess(source_preprocessing)(x))

        x_mini = nn.functional.interpolate(x, (512, 512), mode="bilinear")
        relevance = relevance_model(x_mini)
        relevance = nn.functional.interpolate(relevance, x.shape[2:], mode="bilinear")
        # Discretize to 0 and 1
        relevance = torch.round(relevance)
    else:
        relevance = torch.ones_like(x[:, 0:1])

    return relevance