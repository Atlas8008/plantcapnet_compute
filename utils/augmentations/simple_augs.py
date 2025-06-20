import random

from typing import List

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomRot90(transforms.RandomRotation):
    def __init__(self, interpolation=transforms.InterpolationMode.NEAREST, expand=True, center=None, fill=0):
        super().__init__(0, interpolation, expand, center, fill)

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(random.choice([0, 90, 180, 270]))

        return angle

