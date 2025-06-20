from .utils import ensure_numpy, __as_transform


class ImgaugWrapper:
    def __init__(self, augmentation):
        self.augmentation = augmentation

        self.transform = __as_transform(ensure_numpy(self.augmentation))

    def __call__(self, img):
        return self.transform(img)