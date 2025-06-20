import numpy as np
import torch

from PIL import Image

from utils import math


def image_to_index_map(img, colors, ids=None, target_size=None, single_channel=False):
    """
    Generates a 2D numpy array representing the class index at each pixel from an image.

    :param img: str, numpy.ndarray or PIL Image; the image to convert. If a string is provided, it will be interpreted as a path
    and the image will be loaded from that path first.
    :param colors: A numpy.ndarray of shape (None, 3), containing the corresponding colors for each class id in the
    id parameter.
    :param ids: A 1D numpy.ndarray of the length of the number of classes. The array represents the ids corresponding
    to each color in the colors argument. If None, it will be set to np.arange(0, len(colors)).
    :param target_size: Optional target size for the loaded image. Format: (target_height, target_width).
    :return: An index map, i.e. a 2D numpy array containing the class index at each pixel position.
    """
    if isinstance(img, str):
        annotation_image = Image.open(img)
        annotation_image = np.array(annotation_image)
    elif isinstance(img, np.ndarray):
        annotation_image = img
    else: # PIL Image
        annotation_image = np.array(img)


    if target_size is not None:
        annotation_image = math.resize_interpolate(annotation_image, target_size=target_size, resample="nearest")

    if ids is None:
        ids = np.arange(0, len(colors))

    # Get indices for each color match
    # indices = np.where(
    #     np.all(
    #         np.equal(
    #             annotation_image[None, :],
    #             colors[:, None, None, :]
    #         ),
    #         axis=-1
    #     )
    # )
    # print(indices[0].shape)
    # print(indices[1].shape)
    # print(indices[2].shape)
    #indicesa, indicesb = np.meshgrid(
    #    np.arange(annotation_image.shape[0]), np.arange(annotation_image.shape[1]))
    #indices = np.where()
    #indices = np.zeros(np.prod(annotation_image.shape[:2]), dtype="int32")
    #indices = indicesa, indicesb, indices

    index_image = np.zeros(annotation_image.shape[:2], dtype="int32") - 1

    if single_channel:
        colors = colors[:, 0]

        for idx, color in enumerate(colors):
            index_image[annotation_image[:, :, 0] == color[None, None]] = ids[idx]
    else:
        for idx, color in enumerate(colors):
            #index_image[np.all(annotation_image[:, :] == color[None, None, :], axis=-1)] = ids[idx]
            index_image[
                np.logical_and(
                    np.logical_and(
                        annotation_image[:, :, 0] == color[None, None, 0],
                        annotation_image[:, :, 1] == color[None, None, 1],
                    ),
                    annotation_image[:, :, 2] == color[None, None, 2],
                )
            ] = ids[idx]
    #index_image[indices[1], indices[2]] = ids[indices[0]]

    # Check for -1 values to validate output
    assert -1 not in set(index_image.flatten()), f"Something went wrong, there are pixels with no corresponding " \
                                                 "class color in the annotation image. " + \
                                                 f"(file: {img}" if isinstance(img, str) else ""

    return index_image


def index_map_to_one_hot(index_map, class_count, compressed=False):
    """
    Convert an index map to a one-hot encoded representation.

    This function takes an index map, where each pixel value represents a class index, and converts it to a one-hot encoded representation.

    Parameters:
    - index_map (numpy.ndarray): The input index map with class indices.
    - class_count (int): The total number of classes in the dataset.
    - compressed (bool, optional): If False, the one-hot encoding is created using direct indexing. If True, a compressed representation is used, mapping non-zero indices to a reduced set of indices.

    Returns:
    - tuple: A tuple containing:
        - one_hot_enc (numpy.ndarray): The one-hot encoded representation of the index map.
        - class_unmap (dict or None): If compressed is True, a dictionary mapping compressed indices to original class indices. If compressed is False, None.

    Example:
    ```python
    index_map = np.array([[0, 1], [2, 1]])
    class_count = 3
    one_hot_result, class_unmap = index_map_to_one_hot(index_map, class_count, compressed=True)
    ```
    """
    if not compressed:
        one_hot_enc = np.zeros((*index_map.shape, class_count))
        one_hot_enc[index_map[:, :, None] == np.arange(class_count)[None, None]] = 1

        return one_hot_enc, None
    else:
        non_zeros = set(index_map.flatten())
        class_remap = {c: i for i, c in enumerate(non_zeros)}
        one_hot_enc = np.zeros((*index_map.shape, len(non_zeros)))

        for idx in non_zeros:
            one_hot_enc[index_map == idx, class_remap[idx]] = 1

        class_unmap = {i: c for c, i in class_remap.items()}

        return one_hot_enc, class_unmap

def decompress_one_hot_map(one_hot_enc, class_unmap, n_classes, backend):
    if backend == "numpy":
        full_one_hot = np.zeros((*one_hot_enc.shape[:2], n_classes))

        for i, c in class_unmap.items():
            full_one_hot[..., c] = one_hot_enc[..., i]
    elif backend == "torch":
        full_one_hot = torch.zeros((n_classes, *one_hot_enc.shape[1:]))

        for i, c in class_unmap.items():
            full_one_hot[c, ...] = one_hot_enc[i, ...]
    else:
        raise ValueError("Invalid backend: " + backend)

    return full_one_hot


def one_hot_to_index_map(one_hot):
    """
    Converts a one-hot segmentation map into a 2D index map of maximum indices.

    :param one_hot: A 3D one-hot segmentation map.
    :return: A 2D index map.
    """
    # Get indices of ones and mark unset values with 0
    argmax = np.argmax(one_hot, axis=-1)

    return np.where(np.max(one_hot, axis=-1), argmax, -np.ones_like(argmax))


def index_map_to_image(index_map, colors, ids=None, output_format="img", no_index_color=(0, 0, 0)):
    """
    Converts an index map to an image in numpy or PIL image format.

    :param index_map: A 2D index map.
    :param colors: A 2D array of shape (None, 3) containing a color for each id in ids.
    :param ids: A 1D numpy.ndarray of the length of the number of classes. The array represents the ids corresponding
    to each color in the colors argument. If None, it will be set to np.arange(0, len(colors)).
    :param output_format: The output format of the image. Can be either "img" for PIL image format or "ndarray" for
    numpy array format.
    :param no_index_color: The color value to be used, if there is an unknown index in the index map (value -1).
    :return: The image in the desired format.
    """
    assert output_format in ["img", "ndarray"]

    if ids is None:
        ids = np.arange(0, len(colors))

    indices = np.where(index_map[..., None] == ids[None, None, :])

    img = np.zeros(index_map.shape + (3,))
    img[indices[0], indices[1], :] = colors[indices[2]]

    img[index_map == -1] = np.array(no_index_color)

    if output_format == "img":
        img = Image.fromarray(img.astype("uint8"))

    return img


def discretize_one_hot(one_hot, axis=-1):
    """
    Discretizes a one-hot tensor, i.e. the maximal element will be denoted 1, the rest 0. This can be used to
    discretize the output of a network.

    :param one_hot: A one-hot tensor.
    :param axis: The axis on which to apply the discretization.
    :return: The discretized tensor.
    """
    max_indices = np.argmax(one_hot, axis=axis)

    discretized = np.zeros_like(one_hot)
    discretized[max_indices] = 1

    return discretized