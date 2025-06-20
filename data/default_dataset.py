import os
import torch
import random
import numpy as np

from abc import ABC
from PIL import Image
from torch.utils.data import Dataset
from imgaug import augmenters

from utils.data import ImageServer
from .utils import get_input_label_tuple
from utils.segmentation_utils import image_to_index_map


class DefaultDataset(Dataset, ABC):
    def __init__(self, *, dataset_path="", transform=None, target_transform=None, cache_files=False):
        """Default dataset class for loading images and labels.
        Args:
            dataset_path (str): Path to the dataset folder. Relative paths from the current file are supported.
            transform (callable, optional): Optional transform to be applied on the input image.
            target_transform (callable, optional): Optional transform to be applied on the target.
            cache_files (bool): Whether to cache the input image files in memory.
        """
        self.dataset_path = os.path.join(os.path.split(__file__)[0], dataset_path)
        self.transform = transform
        self.target_transform = target_transform
        self.data_list = None
        self.tuple_order = "pl"
        self.image_server = ImageServer(cache_files=cache_files)
        self.num_classes = -1

    def __len__(self):
        return len(self.data_list)

    def _get_path_label_tuple(self, idx):
        return get_input_label_tuple(self.data_list[idx], self.tuple_order)

    def __getitem__(self, idx):
        path, label = self._get_path_label_tuple(idx)

        image = self.image_server[path]
        #image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_num_classes(self):
        return self.num_classes

    def _reseed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)


class DefaultSegmentationDataset(DefaultDataset):
    def __init__(self, segmentation_path="./", segmentation_resize=None, cache_segmentations=False, apply_default_preprocessing=False, *args, **kwargs):
        """Default dataset class for loading images and labels for segmentation tasks.
        Args:
            segmentation_path (str): Path to the segmentation folder. Relative paths from the current file are supported.
            segmentation_resize (int, optional): Resize the segmentation images to this size. If None, no resizing is done.
            cache_segmentations (bool): Whether to cache the segmentation image files in memory.
            apply_default_preprocessing (bool): Whether to apply default preprocessing to the images.
            *args: Additional arguments for the DefaultDataset class.
            **kwargs: Additional keyword arguments for the DefaultDataset class.
        """
        super().__init__(*args, **kwargs)

        self.segmentation_image_server = ImageServer(cache_files=cache_segmentations)
        self.segmentation_path = segmentation_path

        self.apply_default_preprocessing = apply_default_preprocessing
        self.segmentation_preprocessing = augmenters.Sequential([
            augmenters.Resize(
                {"shorter-side": int(448 / 0.875),
                 "longer-side": "keep-aspect-ratio"}
            ),
            augmenters.CenterCropToMultiplesOf(height_multiple=32, width_multiple=32)
        ])

        if segmentation_resize:
            self.segmentation_resizer = augmenters.Resize(
                {
                    "shorter-side": segmentation_resize,
                    "longer-side": "keep-aspect-ratio"
                },
                interpolation="nearest",
            )
        else:
            self.segmentation_resizer = None

    def __getitem__(self, idx):
        path, label = self._get_path_label_tuple(idx)

        colors = np.array([[i] * 3 for i in range(self.num_classes + 1)])

        image = self.image_server[path]
        label_image = self.segmentation_image_server[label]

        if self.segmentation_resizer:
            label_image = self.segmentation_resizer.augment_image(np.array(label_image))
        segmentation = image_to_index_map(label_image, colors=colors, single_channel=True)

        # Move background index to the back
        segmentation -= 1
        segmentation[segmentation == -1] = self.num_classes
        segmentation = torch.Tensor(segmentation).reshape((1, segmentation.shape[0], segmentation.shape[1]))

        # Save seed for synchronizing image and segmentation augmentations
        seed = torch.seed()
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)

        if self.apply_default_preprocessing:
            image = Image.fromarray(self.segmentation_preprocessing.augment_image(np.asarray(image)))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            # Reset seed for segmentation transform
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            label_image = torch.squeeze(self.target_transform(segmentation).type(torch.LongTensor))

        return image, label_image
