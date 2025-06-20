from imgaug import augmenters
from .utils import ensure_numpy


@ensure_numpy
def pad_to_size(img, size, mode="constant"):
    aug = augmenters.PadToFixedSize(width=size[1], height=size[0], pad_mode=mode)

    return aug.augment_image(img)