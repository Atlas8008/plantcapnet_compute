import os
import json
import torch
import random
import numpy as np

from PIL import Image
from imgaug import augmenters
from torchvision import transforms
from torchvision.transforms import functional as F

from . import DefaultDataset
from utils.segmentation_utils import index_map_to_one_hot, image_to_index_map, decompress_one_hot_map
from utils.data import ImageServer
from utils.augmentations.torchaug import SegmentationBasedInvertedCutout, SegmentationCompose
from utils.augmentations.torchaug import SegmentationBasedZoomIn


class FolderDefaultDict(dict):
    def __missing__(self, key):
        res = f"../datasets/{key}"
        return res


SEGMENTATION_PREPROCESSING_DEFAULT = augmenters.Sequential([
    augmenters.Resize(
        {"shorter-side": int(448 / 0.875),
            "longer-side": "keep-aspect-ratio"}
    ),
    augmenters.CenterCropToMultiplesOf(height_multiple=32, width_multiple=32)
])

def SEGMENTATION_PREPROCESSING_LARGE(img):
    """ Function to recalculate the new image size to be exactly double the one of the small/default resize, as otherwise due to rounding errors they will not match and throw errors. """
    shorter_side = min(img.size)
    longer_side = max(img.size)

    target_shorter_side_small = int(448 / 0.875)
    target_shorter_side_large = target_shorter_side_small * 2

    ratio = target_shorter_side_small / shorter_side

    target_longer_side_small = round(ratio * longer_side)
    target_longer_side_large = target_longer_side_small * 2

    return augmenters.Sequential([
        augmenters.Resize(
            {"shorter-side": target_shorter_side_large,
                "longer-side": target_longer_side_large}
        ),
        augmenters.CenterCropToMultiplesOf(height_multiple=64, width_multiple=64)
    ])

def _find_transforms(o, cls):
    lst = []

    if hasattr(o, "transforms") and isinstance(o.transforms, (list, tuple)):
        for t in o.transforms:
            sublist = _find_transforms(t, cls)

            lst.extend(sublist)

    if isinstance(o, cls):
        lst.append(o)

    return lst


class GBIFDataset(DefaultDataset):
    def __init__(self, split, dataset_path="../datasets/gbif_plants",  segmentation_path=None, include_background_class=False, zoomin_augmentation=None, large_images=False, ic_mask=False, ic_cut_segmap=False, *args, **kwargs):
        assert split in ("train", "validation", "test")

        if isinstance(dataset_path, tuple):
            dataset_path, data_fname = dataset_path
        else:
            data_fname = None

        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.split(__file__)[0], "..", "datasets", dataset_path)

        super().__init__(dataset_path=dataset_path, *args, **kwargs)

        self.segmentation_image_server = ImageServer(cache_files=self.image_server.cache_files)

        if data_fname is not None:
            data_file = os.path.join(self.dataset_path, data_fname)
        else:
            if os.path.exists(os.path.join(self.dataset_path, "data_cleaned.json")):
                data_file = os.path.join(self.dataset_path, "data_cleaned.json")
            else:
                data_file = os.path.join(self.dataset_path, "data.json")

        print("Reading data from", data_file)

        with open(data_file, "r") as f:
            data = json.loads(f.read())

        classes = sorted(data["train"])
        class_to_id_dict = {class_: i for i, class_ in enumerate(classes)}

        print("Classes:")
        print(classes)
        self.compress_segmaps = True
        self.class_names = classes
        self.segmentation_path = segmentation_path
        self.mode = "segmentation" if segmentation_path is not None else "classification"
        self.include_background_class = False if self.mode == "classification" else include_background_class

        self.large_images = large_images

        if large_images:
            self.segmentation_preprocessing = SEGMENTATION_PREPROCESSING_LARGE
        else:
            self.segmentation_preprocessing = SEGMENTATION_PREPROCESSING_DEFAULT

        if split == "train":
            self.data_list = self.extract_tuples(data["train"], class_to_id_dict, mode=self.mode)
        else:
            self.data_list = self.extract_tuples(data["val"], class_to_id_dict, mode=self.mode)

        self.tuple_order = "lp"
        self.num_classes = len(class_to_id_dict) + int(self.include_background_class)

        self.zoomin_augmentation = zoomin_augmentation
        self.ic_mask = ic_mask
        self.ic_cut_segmap = ic_cut_segmap

    def extract_tuples(self, d, class_to_id, mode="classification"):
        assert mode in ("classification", "segmentation")

        tuple_list = []

        for classname in d:
            for imagepath in d[classname]:
                if mode == "classification":
                    tuple_list.append((class_to_id[classname], os.path.join(self.dataset_path, classname, imagepath)))
                else:
                    tuple_list.append((
                        os.path.join(self.segmentation_path, classname, os.path.splitext(imagepath)[0] + ".png"),
                        os.path.join(self.dataset_path, classname, imagepath))
                    )

        return tuple_list

    def get_image_and_segmentation(self, img_path, segmap_path, compress_segmap=False):

        #colors = np.array([[i] * 3 for i in range(len(self.class_names) + 1)])
        colors = np.stack([(np.array([1, 1, 1]) * i + [i // 256, 0, 0]) % 256 for i in range(len(self.class_names) + 1)], axis=0) # Added shift to support more than 256 classes

        image = self.image_server[img_path] # Image.open(path).convert("RGB")
        label_image = self.segmentation_image_server[segmap_path]

        if hasattr(self.segmentation_preprocessing, "__call__") and not hasattr(self.segmentation_preprocessing, "augment_image"):
            segmentation_preprocessor = self.segmentation_preprocessing(image)
        else:
            segmentation_preprocessor = self.segmentation_preprocessing

        image = Image.fromarray(segmentation_preprocessor.augment_image(np.asarray(image)))

        segmentation = image_to_index_map(label_image, colors=colors)
        segmentation, class_unmap = index_map_to_one_hot(
            segmentation, len(self.class_names) + 1, compressed=compress_segmap)

        return image, segmentation, class_unmap

    #@profile_fun
    def __getitem__(self, idx):
        path, label = self._get_path_label_tuple(idx)

        #image_transform = deepcopy(self.transform)
        #target_transform = deepcopy(self.target_transform)

        image_transform = self.transform
        target_transform = self.target_transform

        if self.mode == "classification":
            image = self.image_server[path] # Image.open(path).convert("RGB")

            if image_transform:
                image = image_transform(image)
            if target_transform:
                label = target_transform(label)
        else:
            image, segmentation, class_unmap = self.get_image_and_segmentation(path, label, compress_segmap=self.compress_segmaps)

            if self.large_images:
                segmentation = np.repeat(segmentation, 2, axis=0).repeat(2, axis=1)
                target_size = (image.size[0] // 2, image.size[1] // 2)
            else:
                target_size = image.size

            if self.zoomin_augmentation is not None:
                image, segmentation = self.zoomin_augmentation(
                    image, segmentation, target_size=target_size)

            image, label, final_mask = self.apply_transforms_functional(
                image,
                segmentation,
                image_transform,
                target_transform,
                idx,
            )

            if self.ic_mask:
                if final_mask is None:
                    final_mask = torch.ones_like(image[0:1], dtype=torch.float32)

                return image, label, final_mask

            if self.compress_segmaps:
                if not self.include_background_class:
                    class_unmap = {i - 1: c - 1 for i, c in class_unmap.items() if i > 0}
                    class_count = len(self.class_names)
                else:
                    class_count = len(self.class_names) - 1

                label = decompress_one_hot_map(
                    label,
                    class_unmap,
                    class_count,
                    backend="torch",
                )

        return image, label

    def apply_transforms_functional(self, image, segmentation, image_transform, target_transform, idx):

        seed = torch.seed()

        # Remove background class
        if not self.include_background_class:
            segmentation = segmentation[:, :, 1:]
            segmentation_wo_bg = segmentation

        # Transform segmentations first
        if target_transform:
            self._reseed(seed)
            label = target_transform(segmentation).to(torch.float32)

        self._reseed(seed)

        guiding_segmap = label.numpy()
        guiding_segmap = np.moveaxis(guiding_segmap, 0, -1)

        if self.include_background_class:
            guiding_segmap = guiding_segmap[:, :, 1:]

        final_mask = None

        if image_transform:
            if isinstance(image_transform, SegmentationCompose):
                image, addtl_outputs = image_transform(image, guiding_segmap)

            else:
                image = image_transform(image)
                addtl_outputs = {}

            masks = []

            if "IC valid mask" in addtl_outputs:
                masks.append(
                    torch.tensor(
                        1 - addtl_outputs["IC valid mask"]
                    )
                )

                if self.ic_cut_segmap:
                    m = torch.tensor(
                        addtl_outputs["IC valid mask"]
                    )
                    m = torch.moveaxis(m, -1, 0)
                    label = label * m

            if masks:
                final_mask = masks[0]

                for mask in masks[1:]:
                    final_mask = torch.logical_or(final_mask, mask)

                if len(final_mask.shape) == 3:
                    final_mask = torch.moveaxis(final_mask, -1, 0)

                final_mask = final_mask.to(torch.float32)

        return image, label, final_mask


    def _apply_transforms_stateful(self, image, segmentation, image_transform, target_transform, idx):

        seed = torch.seed()

        # Remove background class
        if not self.include_background_class:
            bgs = segmentation[:, :, 0:1]
            segmentation = segmentation[:, :, 1:]

        # Transform segmentations first
        if target_transform:
            self._reseed(seed)
            label = target_transform(segmentation).to(torch.float32)

        ic_augs = _find_transforms(image_transform, SegmentationBasedInvertedCutout)

        # Handling of inverted cutout
        contains_ic = len(ic_augs) > 0

        if contains_ic:
            guiding_segmap = label.numpy()
            guiding_segmap = np.moveaxis(guiding_segmap, 0, -1)

            # Remove background class for IC sampling if used
            if self.include_background_class:
                guiding_segmap = guiding_segmap[:, :, 1:]

            for transform in ic_augs:
                transform.sample_location_from_segmap(guiding_segmap)

                if transform.segmix:
                    label = self.apply_segmix(transform, label)

        self._reseed(seed)

        if image_transform:
            image = image_transform(image)

            masks = []

            for transform in ic_augs:
                # Maybe apply ic based crop to segmentations
                label = transform.maybe_crop_segmentation(label)

                if transform._last_valid_map is not None:
                    masks.append(torch.tensor(transform.invalid_mask))

            final_mask = None

            if masks:
                final_mask = masks[0]

                for mask in masks[1:]:
                    final_mask = torch.logical_or(final_mask, mask)

                if len(final_mask.shape) == 3:
                    final_mask = torch.moveaxis(final_mask, -1, 0)

                final_mask = final_mask.to(torch.float32)

        return image, label, final_mask

    def classwise_segmentation_volumes(self):
        volumes = None

        for i in range(len(self)):
            path, label = self._get_path_label_tuple(i)
            image, segmentation = self.get_image_and_segmentation(path, label)

            if volumes is None:
                volumes = np.sum(segmentation, axis=(0, 1))
            else:
                volumes += np.sum(segmentation, axis=(0, 1))


        if not self.include_background_class:
            volumes = volumes[:-1]

        return volumes


    def get_num_classes(self):
        return self.num_classes

    def apply_segmix(self, transform, label):
        # Sample new image and segmap
        segmix_img_path, segmix_segmap_label = \
            self._get_path_label_tuple(
                random.randrange(0, len(self.data_list)))
        segmix_img, segmix_segmap = \
            self.get_image_and_segmentation(
                segmix_img_path,
                segmix_segmap_label
        )

        if not self.include_background_class:
            segmix_segmap = segmix_segmap[:, :, 1:]

        segmix_img, segmix_segmap = transform.provide_and_transform_segmix_image_and_segmentation(segmix_img, segmix_segmap)
        segmix_segmap = segmix_segmap.to(torch.float32)

        label = torch.where(
            label != 0,
            label,
            segmix_segmap,
        )

        return label


class ImageZoomIn:
    def __init__(self, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), interpolation=transforms.InterpolationMode.BILINEAR, sampling_remove_index=None) -> None:
        self.zoomin_transform = SegmentationBasedZoomIn(
            size=(224, 224),
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
        )

        self.sampling_remove_index = sampling_remove_index

    def __call__(self, img, segmentation, target_size=None):
        img_size = tuple(F.get_image_size(img))
        segm_size = segmentation.shape[:2][::-1]
        assert img_size == segm_size, f"Image and segmentation map should have the same size, were {img_size} and {segm_size}"

        if self.sampling_remove_index is not None:
            segmentation_sampling = np.concatenate([
                segmentation[..., :self.sampling_remove_index],
                segmentation[..., self.sampling_remove_index + 1:],
            ])
        else:
            segmentation_sampling = segmentation

        self.zoomin_transform.sample_parameters(img, segmentation_sampling)

        # Set size to original image size
        if target_size is None:
            self.zoomin_transform.size = img_size[::-1]
        else:
            self.zoomin_transform.size = target_size

        img = self.zoomin_transform(
            img,
            interpolation=transforms.InterpolationMode.BILINEAR,
            clear_params=False
        )
        segmentation = self.zoomin_transform(
            segmentation,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        return img, segmentation
