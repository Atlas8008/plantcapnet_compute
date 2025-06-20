import re

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.augmentations.jigsaw import AdvancedRandomJigsaw
from utils.augmentations import RandomRot90


class RandomScaleResize(nn.Module):
    def __init__(self, min_scale, image_size):
        super().__init__()
        self.min_scale = min_scale
        self.image_size = image_size

    def forward(self, img):
        scale = self.min_scale + (1.0 - self.min_scale) * torch.rand(1).item()
        scaled_size = [int(dim * scale) for dim in self.image_size]
        img = F.resize(img, scaled_size, interpolation=transforms.InterpolationMode.BICUBIC)
        img = F.resize(img, self.image_size, interpolation=transforms.InterpolationMode.BICUBIC)
        return img


def get_augmentations(aug_spec, image_size=None):
    """
    Get augmentations based on the specified augmentation strategy.

    Args:
        aug_spec (str): Augmentation specification string. The format is a
        comma-separated list of augmentation strategies, where each strategy
        can be one of the following:
            - "h" or "hflip": Horizontal flip.
            - "v" or "vflip": Vertical flip.
            - "p" or "perspective": Perspective transformation.
            - "rot90": Random 90-degree rotations.
            - "jWxH[hvrs]": Advanced jigsaw with grid size WxH and optional
              parameters:
                - "h": Random horizontal flip.
                - "v": Random vertical flip.
                - "r": Random rotation.
                - "s": Shift.
            - "rPeAsS": Elastic transform with probability P, alpha A, and sigma S.
            - "dP": Dropout with probability P.
            - "lrS": Low resolution with minimum scale S.
        image_size (tuple, optional): Image size for random scale resize augmentation.
    """
    start_augmentations = []
    mid_augmentations = []
    final_augmentations = []

    spec_conversions = {
        "hflip": "h",
        "hvflip": "h,v",
        "hflip_delpx25": "h,d0.25",
        "hflip_delpx50": "h,d0.5",
        "hflip_elastic50": "h,r0.5e50s5",
        "hflip_elastic25": "h,r0.5e25s5",
        "hflip_perspective": "h,p",
    }

    if aug_spec in spec_conversions:
        aug_spec = spec_conversions[aug_spec]

    augs = aug_spec.split(",")

    for aug in augs:
        if aug in ("h", "hflip"):
            mid_augmentations.append(transforms.RandomHorizontalFlip())
        elif aug in ("v", "vflip"):
            mid_augmentations.append(transforms.RandomVerticalFlip())
        elif aug in ("p", "perspective"):
            mid_augmentations.append(transforms.RandomPerspective(
                distortion_scale=0.85))
        elif aug == "rot90":
            mid_augmentations.append(RandomRot90())
        elif aug.startswith("j"):
            params = aug[1:]

            random_hflip = False
            random_vflip = False
            random_rotate = False
            shift = False

            while params:
                if match := re.match(r"([0-9]+)x([0-9]+)", params):
                    grid_size = (int(match.group(1)), int(match.group(2)))
                    params = params[len(match.group(0)):]
                else:
                    p = params[0]
                    params = params[1:]

                    if p == "h":
                        random_hflip = True
                    elif p == "v":
                        random_vflip = True
                    elif p == "r":
                        random_rotate = True
                    elif p == "s":
                        shift = True
                    else:
                        raise ValueError("Invalid parameter: " + p)

            final_augmentations.append(
                AdvancedRandomJigsaw(
                    grid=grid_size,
                    data_keys=["image"],
                    p=1.0,
                    random_hflip=random_hflip,
                    random_vflip=random_vflip,
                    random_rotate=random_rotate,
                    shift=shift,
                    keepdim=True,
                )
            )
        elif match := re.match(r"r([0-9.]+)ea([0-9.]+)s([0-9.]+)", aug): # Elastic transform augmentation
            p = float(match.group(1))
            alpha = float(match.group(2))
            sigma = float(match.group(3))

            mid_augmentations.append(
                transforms.RandomApply([
                    transforms.ElasticTransform(alpha=alpha, sigma=sigma),
            ], p=p))
        elif aug.startswith("d"): # Dropout augmentation
            p = float(aug[1:])

            final_augmentations.append(
                transforms.Lambda(lambda x: nn.functional.dropout(x, p=p)))
        elif aug.startswith("lr"): # Low resolution augmentation
            assert image_size is not None, "Image size must be provided for random scale resize augmentation"

            min_scale = float(aug[2:])

            t = transforms.RandomApply([
                RandomScaleResize(min_scale, image_size)
            ], p=0.5)

            mid_augmentations.append(t)
        else:
            raise ValueError(f"Invalid aug strat: {aug_spec}, {aug} is not a valid augmentation")

    return start_augmentations, mid_augmentations, final_augmentations