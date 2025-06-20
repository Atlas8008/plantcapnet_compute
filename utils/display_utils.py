import os
import matplotlib
import numpy as np

if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.colors import ListedColormap


def image_as_matrix(img_or_path, dtype=None, reshape_three_channel=True, size=None, min_dim=None, return_scale=False):
    """
    Given a path: reads an image from file and returns it as numpy array.
    Given a PIL image: converts the image into a numpy array and returns it.

    :param img_or_path: An image or a filepath.
    :param dtype: The target type of the array. If None, the default value will be used.
    :param reshape_three_channel: Bool, if True, the image will be read and reshaped to contain only three
    channels, if, for example, an additional alpha channel is contained in the data.
    :param size: A tuple containing the target width and height of the image or numpy array respectively.
    :param min_dim: If a value is provided, the image will be scale so that the smaller side's length is equal to
    min_dim. Value is ignored, if size is provided.
    :param return_scale: Bool; if True, the scale in comparison to the original image dimensions will be returned
    together with the image numpy array. This is a tuple (x_scale, y_scale).
    :return: The image as numpy array.
    """
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path, "r")
    else:
        img = img_or_path

    if size is None:
        size = img.size

        # Resize to minimal dimension
        if min_dim is not None:
            min_val = min(size)

            # Only, if the smaller value is greater than min_dim
            if min_dim < min_val:
                idx = size.index(min_val)

                if idx == 0:
                    size = (min_dim, int((min_dim / min_val) * size[1]))
                else:
                    size = (int((min_dim / min_val) * size[0]), min_dim)

    scale = size[0] / img.size[0], size[1] / img.size[1]

    if size is not None:
        img = img.resize(size, resample=Image.BILINEAR)

    img_array = np.asarray(img)

    if reshape_three_channel and img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]

    if dtype is not None:
        img_array = img_array.astype(dtype)

    if return_scale:
        return img_array, scale

    return img_array


def display_image(img_or_ndarray, do_show=True, title=None, **kwargs):
    if not isinstance(img_or_ndarray, np.ndarray):
        ary = np.asarray(img_or_ndarray)
    else:
        ary = img_or_ndarray

    return _default_plot(ary, do_show, title=title, **kwargs)


def display_tabular(img_list, format, do_show=True, title=None):
    fig = plt.figure(figsize=(12, 7))

    for i in range(format[0]):
        for j in range(format[1]):
            if i * format[1] + j >= len(img_list):
                break

            plt.subplot(format[0], format[1], i * format[1] + j + 1)
            plt.axis("off")
            plt.tight_layout()

            plt.imshow(img_list[i * format[1] + j])

    if title is not None:
        fig.canvas.set_window_title(title)

    if do_show:
        plt.show()

    return fig


def display_heatmap(array, title=None):
    return _default_plot(array, title=title)


def display_overlay(overlay, image, overlay_color_rgb=(255, 0, 0), overlay_gradient_lower_color_rgb=None, gray_image=True, title="", do_show=True, save_to=""):
    """
    Uses a 2D numpy array and displays it as an overlay over a provided image. The values in the overlay will be
    normalized between their min and max values. If the overlay does not have the same size as the provided image, it
    will be resized to the required size using nearest neighbor interpolation.

    :param overlay: A 2d numpy array.
    :param image: The image, over which the overlay will be laid.
    :param overlay_color_rgb: The color of the maximum values in the overlay map.
    :param gray_image: Bool; designates, if the original image will be shown in grayscale or full RGB.
    :param title: The title of the matplotlib plot.
    :param do_show: Bool; designates, if the plot will be shown directly before returning.
    :param save_to: An optional path for saving the plot.
    :return: The matplotlib figure containing the overlaid image.
    """
    # Normalize the overlay to values between 0 and 1
    min_, max_ = np.min(overlay), np.max(overlay)
    overlay = (overlay - min_) / (max_ - min_)

    # Remove empty dimensions from overlay
    overlay = np.squeeze(overlay)

    # Convert overlay to 3-channel image
    if len(overlay.shape) < 3:
        if isinstance(overlay_color_rgb, str):  # This means, as colormap with a name was selected
            overlay = np.stack([
                    overlay * 255,
                    overlay * 255,
                    overlay * 255,
                    127 * overlay  # Transparency
                ],
                axis=-1
            )
        elif overlay_gradient_lower_color_rgb is None:
            overlay = np.stack([
                    overlay * overlay_color_rgb[0],
                    overlay * overlay_color_rgb[1],
                    overlay * overlay_color_rgb[2],
                    255 * overlay # Transparency
                ],
                axis=-1
            )
        else:
            overlay = np.stack([
                    overlay * overlay_color_rgb[0] + (1 - overlay) * overlay_gradient_lower_color_rgb[0],
                    overlay * overlay_color_rgb[1] + (1 - overlay) * overlay_gradient_lower_color_rgb[1],
                    overlay * overlay_color_rgb[2] + (1 - overlay) * overlay_gradient_lower_color_rgb[2],
                    127 * np.ones_like(overlay)  # Transparency
                ],
                axis=-1
            )

    overlay = overlay.astype("uint8")

    # Convert overlay and image to PIL images
    overlay = Image.fromarray(overlay)

    if isinstance(image, str):
        image = Image.open(image)

        if gray_image:
            image = image.convert("L")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))

    if overlay.size != image.size:
        overlay = overlay.resize(image.size)

    fig = plt.figure(figsize=(14, 7))

    display_image(image, title=title, do_show=False, alpha=1, new_fig=False, cmap="gray" if gray_image else None)
    if not isinstance(overlay_color_rgb, str):
        plt.imshow(overlay, alpha=1)
    else:
        colormap = plt.cm.get_cmap(overlay_color_rgb)
        plt.imshow(np.array(overlay)[..., 0], cmap=colormap, alpha=0.5)

    if save_to:
        plt.savefig(save_to, bbox_inches="tight", transparent=True)

    if do_show:
        plt.show()

    return fig


def display_segmentation(segmentation_map, image, class_name_list, *, colormap=None, title="", do_show=True, save_to="",
                         colorbar=True, close_fig=False):
    """
    Renders a segmentation map onto an image for evaluation of the goodness of the segmentation.

    :param segmentation_map: A 3D one-hot encoded segmentation map or a 2D map, which contains the index of the
    predicted class in each position. The actual input is inferred automatically.
    :param image: Either a string, designating the path to an image, or a PIL image object.
    :param class_name_list: The list of classes to display.
    :param colormap: An optional numpy array of shape (len(class_name_list), 3), representing the color for each class
    in the class name list. If set to None, a rainbow color map will be used. Colors should be in the range of (0,1).
    :param title: The title of the plot window.
    :param do_show: Designates, if the plot will be shown directly.
    :param save_to: Optional path to save the plot to.
    :return: The created figure.
    """
    # Load image and/or convert to grayscale
    if isinstance(image, str):
        image = Image.open(image).convert("L")
        img_ary = np.array(image)
        imagesize = image.size
    elif isinstance(image, np.ndarray):
        #img_ary = image
        image = Image.fromarray(image).convert("L")
        img_ary = np.array(image)
        #imagesize = img_ary.shape[:2][::-1]
        imagesize = image.size
    else:
        image = image.convert("L")
        img_ary = np.array(image)
        imagesize = image.size

    # Potentially convert 3D one-hot map to 2D index map
    if len(segmentation_map.shape) == 3:
        index_map = np.argmax(segmentation_map, axis=-1)
    else:
        index_map = segmentation_map

    #img_ary = np.array(image)

    index_map_img = Image.fromarray(index_map.astype("uint8"))
    index_map_img = index_map_img.resize(imagesize)

    index_map = np.array(index_map_img)

    if colormap is None:
        colormap = plt.cm.get_cmap("gist_rainbow_r", len(class_name_list))
    elif isinstance(colormap, str):
        colormap = plt.cm.get_cmap(colormap, len(class_name_list))
    else:
        ones = np.ones((len(colormap), 1))

        colormap = np.concatenate([colormap, ones], axis=-1)
        colormap = ListedColormap(colormap)

    fig = plt.figure(figsize=(14, 7))

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Draw colormap of segmentations
    plot = plt.imshow(index_map, cmap=colormap, alpha=1, vmax=len(class_name_list) - 1)

    if colorbar:
        # Configure color bar
        cbar = plt.colorbar(plot, pad=0.005, shrink=0.9)
        cbar.ax.get_yaxis().set_ticks([])

    display_image(img_ary, title=title, do_show=False, alpha=0.75, new_fig=False, cmap="gray")

    if colorbar:
        # Add labels to each color
        for i, lbl in enumerate(class_name_list):
            cbar.ax.text(1.2, (i + 0.5) / len(class_name_list), lbl, va="center", transform=cbar.ax.transAxes, fontdict={"size": 14})

    plt.tight_layout()

    if save_to:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(save_to, bbox_inches="tight", transparent=True)

    if do_show:
        plt.show()

    if close_fig:
        plt.close()

    return fig


def _default_plot(array, do_show=True, title=None, new_fig=True, close=False, **kwargs):
    if new_fig:
        fig = plt.figure(figsize=(12, 7))
    else:
        fig = plt.gcf()

    plt.axis("off")
    plt.tight_layout()
    plt.imshow(array, **kwargs)

    if title is not None and do_show:
        plt.gcf().canvas.set_window_title(title)

    if do_show:
        plt.show()

    if close:
        plt.close()

    return fig
