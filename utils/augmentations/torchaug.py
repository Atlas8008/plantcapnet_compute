import torch
import random
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional as F

from utils.augmentations.utils import ensure_numpy

if __name__ == "__main__":
    print("Added superpath")
    import sys; sys.path.append("..")

    from cutout import _apply_cutout, _sample_mask_size
    from utils import method_ensure_numpy
else:
    from utils.augmentations.cutout import _apply_cutout, _sample_mask_size
    from utils.augmentations.utils import method_ensure_numpy

__all__ = ["RandomResize", "SegmentationBasedInvertedCutout"]


class RandomResize(transforms.Resize):
    def __init__(self, scale, interpolation=..., max_size=None, antialias=None):
        size = 512
        super().__init__(size, interpolation, max_size, antialias)

        assert isinstance(scale, float) or isinstance(scale, (tuple, list)) and len(scale) == 2, "Scale should be either a float value or a tuple containing a min and a max value."

        self.scale = scale

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        w, h = F.get_image_size(img)

        if isinstance(self.scale, (tuple, list)):
            scale = (self.scale[1] - self.scale[0]) * torch.rand(1) + self.scale[0]
        else:
            scale = self.scale

        w = int(scale * w)
        h = int(scale * h)

        return F.resize(img, (h, w), self.interpolation, self.max_size, self.antialias)

class SegmentationBasedInvertedCutoutFunctional(torch.nn.Module):
    def __init__(self, min_mask_size: int, max_mask_size: int, independent_wh=False, reps=1, circular=False, seg_visible_crop=False, segmix=False, segmix_img_transform=None, segmix_segmap_transform=None) -> None:
        super().__init__()

        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.independent_wh = independent_wh
        self.reps = reps
        self.circular = circular

        self.segmix = segmix
        self.segmix_tuple = None
        self.segmix_img_transform = segmix_img_transform
        self.segmix_segmap_transform = segmix_segmap_transform

        self.seg_visible_crop = seg_visible_crop

        self.loc = None
        self._last_valid_map = None

    def _sample_location_from_segmap(self, segmentation):
        if len(segmentation.shape) == 3:
            segmentation = np.sum(segmentation, axis=2)

        locations = list(zip(*np.nonzero(segmentation)))

        # As fallback, if no segmentations are contained in the current image part, sample from all other locations
        if len(locations) == 0:
            locations = list(zip(*np.nonzero(segmentation == 0)))

        return [random.choice(locations) for _ in range(self.reps)]

    def _segmix_transform(self, image, segmentation):
        torch_state = torch.random.get_rng_state()
        random_state = random.getstate()
        numpy_state = np.random.get_state()

        if self.segmix_img_transform is not None:
            image = self.segmix_img_transform(image)

        torch.random.set_rng_state(torch_state)
        random.setstate(random_state)
        np.random.set_state(numpy_state)

        if self.segmix_segmap_transform is not None:
            segmentation = self.segmix_segmap_transform(segmentation)

        return image, segmentation

    def maybe_crop_segmentation(self, segmentation, valid):
        if self.seg_visible_crop and valid is not None:
            return self._component_view_based_segmentation_crop(
                valid,
                segmentation,
            )

        return segmentation

    def _component_view_based_segmentation_crop(self, region_mask, segmentation):
        from scipy.ndimage.measurements import label

        is_tensor = isinstance(segmentation, torch.Tensor)
        dtype = None

        if is_tensor:
            dtype = segmentation.dtype
            segmentation = torch.moveaxis(segmentation, 0, -1).cpu().numpy().astype("int32")

        if len(segmentation.shape) == 3:
            bin_segmentation = np.any(segmentation, axis=-1)
        else:
            bin_segmentation = segmentation

        labels, num_feats = label(bin_segmentation)
        region_mask = np.squeeze(region_mask)
        region_labels = region_mask * labels

        # plt.figure(figsize=(20, 10))
        # plt.subplot(121)
        # plt.imshow(labels.astype("uint8"), cmap="nipy_spectral")
        # plt.axis("off")
        # plt.subplot(122)
        # plt.imshow(region_labels.astype("uint8"), cmap="nipy_spectral")
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig(debug_path)

        valid_labels = set(region_labels.flatten().tolist())

        valid_segmentations = np.any(
            [labels == valid_label for valid_label in valid_labels],
            axis=0,
        )

        segmentation = np.where(
            valid_segmentations[..., None],
            segmentation,
            np.zeros_like(segmentation),
        )

        if is_tensor:
            segmentation = torch.moveaxis(
                torch.tensor(segmentation, dtype=dtype), -1, 0)

        return segmentation

    @method_ensure_numpy
    def forward(self, img, segmentation):
        loc = self._sample_location_from_segmap(segmentation)
        if self.segmix:
            segmix_tuple = self._segmix_transform(img, segmentation)

        valid = np.ones_like(img[..., 0:1])

        for loc in loc:
            y, x = loc

            mask_size_x, mask_size_y = _sample_mask_size(
                self.min_mask_size,
                self.max_mask_size,
                self.independent_wh,
            )

            valid = _apply_cutout(
                valid,
                x=x,
                y=y,
                mask_size_x=mask_size_x,
                mask_size_y=mask_size_y,
                circular=self.circular,
            )

        # Invert to cutout only the unselected parts
        valid = 1 - valid

        if self.segmix:
            img2, segmap2 = segmix_tuple

            img = valid * img + (1 - valid) * img2
        else:
            img = valid * img

        return img, {"IC valid mask": valid} # Valid contains ones at the locations where the image parts have been kept by inverted cutout

class SegmentationBasedInvertedCutout(SegmentationBasedInvertedCutoutFunctional):
    def __init__(self, min_mask_size: int, max_mask_size: int, independent_wh=False, reps=1, circular=False, seg_visible_crop=False, segmix=False, segmix_img_transform=None, segmix_segmap_transform=None) -> None:
        super().__init__(min_mask_size, max_mask_size, independent_wh, reps, circular, seg_visible_crop, segmix, segmix_img_transform, segmix_segmap_transform)

        self.loc = None
        self._last_valid_map = None

    def sample_location_from_segmap(self, segmentation):
        self.loc = self._sample_location_from_segmap(segmentation)
        self._last_valid_map = None

    def provide_and_transform_segmix_image_and_segmentation(self, image, segmentation):

        self.segmix_tuple = self._segmix_transform(image, segmentation)

        return self.segmix_tuple

    def maybe_crop_segmentation(self, segmentation):
        if self.seg_visible_crop and self._last_valid_map is not None:
            return self._component_view_based_segmentation_crop(
                self._last_valid_map,
                segmentation,
            )

        return segmentation

    @property
    def valid_mask(self):
        return self._last_valid_map

    @property
    def invalid_mask(self):
        return 1 - self._last_valid_map

    @method_ensure_numpy
    def forward(self, img):
        assert self.loc is not None, "No location sampled; call sample_location_from_segmap with a segmentation map first to sample a location."
        if self.segmix:
            assert self.segmix_tuple is not None, "No image provided for segmix; call provide_segmix_image_and_segmentation first."

        valid = np.ones_like(img[..., 0:1])

        for loc in self.loc:
            y, x = loc

            mask_size_x, mask_size_y = _sample_mask_size(
                self.min_mask_size,
                self.max_mask_size,
                self.independent_wh,
            )

            valid = _apply_cutout(
                valid,
                x=x,
                y=y,
                mask_size_x=mask_size_x,
                mask_size_y=mask_size_y,
                circular=self.circular,
            )

        # Invert to cutout only the unselected parts
        valid = 1 - valid

        if self.segmix:
            img2, segmap2 = self.segmix_tuple

            img = valid * img + (1 - valid) * img2
        else:
            img = valid * img

        self._last_valid_map = valid
        self.loc = None

        return img




class SegmentationBasedZoomIn(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BILINEAR) -> None:
        super().__init__(
            size=size,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )

        self.params = None

    def sample_parameters(self, img, segmentation):
        if len(segmentation.shape) == 3:
            segmentation = np.sum(segmentation, axis=2)

        locations = list(zip(*np.nonzero(segmentation)))

        # As fallback, if no segmentations are contained in the current image part, sample from all other locations
        if len(locations) == 0:
            locations = list(zip(*np.nonzero(segmentation == 0)))

        loc = random.choice(locations)

        img_w, img_h = F.get_image_size(img)

        y, x, h, w = self.get_params(img, self.scale, self.ratio)

        # Overwrite y and x with the preselected location
        y, x = loc

        # Ensure that the crop lies inside the image
        if y + h // 2 >= img_h:
            y -= (y + h // 2 + 1) - img_h
        elif y - h // 2 < 0:
            y -= y - h // 2

        if x + w // 2 >= img_w:
            x -= (x + w // 2 + 1) - img_w
        elif x - w // 2 < 0:
            x -=  x - w // 2

        # Top-left-align coords
        y -= h // 2
        x -= w // 2

        self.params = (y, x, h, w)

    def forward(self, img, interpolation=None, clear_params=True):
        assert self.params is not None, "No parameters sampled; call sample_parameters first to sample parameters."

        if interpolation is None:
            interpolation = self.interpolation

        y, x, h, w = self.params

        if isinstance(img, np.ndarray):
            wasarray = True

            img = torch.tensor(img)
            img = torch.moveaxis(img, -1, 0)
        else:
            wasarray = False

        out = F.resized_crop(img, y, x, h, w, self.size, interpolation)

        if wasarray:
            out = torch.moveaxis(out, 0, -1)
            out = out.numpy()

        if clear_params:
            self.params = None

        return out


class SegmentationCompose(transforms.Compose):
    def __init__(self, transforms, collect_additional_outputs=False):
        super().__init__(transforms)
        #self.transforms = transforms
        self.collect_additional_outputs = collect_additional_outputs

    def __call__(self, img, segmentation):
        additional_outputs = {}

        for transform in self.transforms:
            transform, w_segmentation = get_transform(transform)

            if w_segmentation:
                img = transform(img, segmentation)
            else:
                img = transform(img)

            if isinstance(img, tuple):
                add_outputs = img[1]
                img = img[0]

                additional_outputs.update(add_outputs)

        if self.collect_additional_outputs:
            return img, additional_outputs
        else:
            return img


class SegmentationCutout:
    def __init__(self, ignore_index=None) -> None:
        self.ignore_index = ignore_index

    @method_ensure_numpy
    def __call__(self, img, segmentation):
        if self.ignore_index is not None:
            segmentation = np.concatenate([
                segmentation[:self.ignore_index],
                segmentation[self.ignore_index + 1:]
            ])

        segmentation = np.max(segmentation, axis=-1, keepdims=True)

        return segmentation * img

class RandomChoiceImg(transforms.RandomChoice):
    def __call__(self, img, *args):
        t = random.choices(self.transforms, weights=self.p)[0]

        transform, w_segmentation = get_transform(t)

        if w_segmentation:
            img = transform(img, *args)
        else:
            img = transform(img)

        return img


def get_transform(transform):
    if isinstance(transform, dict):
        assert len(transform) == 1

        k = list(transform.keys())[0]

        transform = transform[k]

        if k == "with segmentation":
            return transform, True
        elif k == "without segmentation":
            return transform, False
        else:
            raise ValueError("Key should be either 'with segmentation' or 'without segmentation'.")
    else:
        return transform, False
