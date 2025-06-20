import numpy as np

from PIL import Image
from functools import wraps
from torchvision import transforms


def __as_transform(fn):
    def fn_transform(*args, **kwargs):
        def transform_fun(img):
            return fn(img, *args, **kwargs)

        return transforms.Lambda(transform_fun)
    return fn_transform


def ensure_numpy(fn):
    @wraps(fn)
    def numpyd_fn(img, *args, **kwargs):
        is_image = isinstance(img, Image.Image)

        if is_image:
            img = np.array(img)

        img = fn(img, *args, **kwargs)

        if isinstance(img, tuple):
            rest = img[1:]
            img = img[0]
        else:
            rest = tuple()

        if is_image:
            img = Image.fromarray(img.astype("uint8"))

        if len(rest) > 0:
            return img, *rest
        else:
            return img

    return numpyd_fn


def method_ensure_numpy(fn):
    @wraps(fn)
    def numpyd_fn(self, img, *args, **kwargs):
        is_image = isinstance(img, Image.Image)

        if is_image:
            img = np.array(img)

        img = fn(self, img, *args, **kwargs)

        if isinstance(img, tuple):
            rest = img[1:]
            img = img[0]
        else:
            rest = tuple()

        if is_image:
            img = Image.fromarray(img.astype("uint8"))

        if len(rest) > 0:
            return img, *rest
        else:
            return img

    return numpyd_fn