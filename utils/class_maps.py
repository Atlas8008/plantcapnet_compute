import os
import torch
import platform
import matplotlib
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from utils import math, display_utils as display


def generate_class_maps(
        model,
        input_image_transform,
        class_names,
        device,
        *,
        input_images=None,
        output_index=0,
        file_extension="jpg",
        do_show=True,
        save_folder=None,
        discretize=True,
        relative=False,
        heatmap_per_channel=False,
        model_mode="segmentation",
        **model_kwargs,
    ):

    if model_kwargs is None:
        model_kwargs = {}

    if model_mode is not None:
        model_kwargs["mode"] = model_mode

    model = model.to(device)

    for image_idx, input_image_path in enumerate(input_images):
        # Predict output
        img = Image.open(input_image_path)
        x = input_image_transform(img)[None, ...]

        with torch.amp.autocast(device.type), torch.no_grad():
            prediction_full = model(
                x.to(device), **model_kwargs)
            if isinstance(prediction_full, (tuple, list)):
                prediction_full = prediction_full[output_index]
            prediction = prediction_full[0]
            prediction = torch.moveaxis(prediction, 0, -1).cpu().numpy()

        print(prediction.shape)
        print(img.size)

        # Resize
        prediction = math.resize_interpolate(prediction, img.size[:2][::-1])

        # Generate segmentation map --> discretize values
        if discretize:
            prediction_one_hot = np.zeros_like(prediction)
            prediction_one_hot[np.max(prediction, axis=-1, keepdims=True) == prediction] = 1

            prediction = prediction_one_hot
        elif relative:
            prediction = prediction / np.max(prediction, axis=(-3, -2), keepdims=True)

        # Clip values to between 0 and 1
        prediction = np.clip(prediction, 0, 1)

        # Draw every heatmap/segmentation map on its own
        if heatmap_per_channel:
            prediction_all_channels = prediction
            # Iterate all channels of the prediction
            for i in range(prediction_all_channels.shape[2]):
                prediction = prediction_all_channels[:, :, i]

                title = "({}) {}  -  ".format(image_idx + 1, input_image_path) + class_names[i]

                if save_folder is not None:
                    save_path = os.path.join(save_folder, "output_img{}_{}.jpg".format(image_idx, class_names[i]))
                else:
                    save_path = None

                single_channel_heatmap(
                    prediction,
                    orig_image=img,
                    title=title,
                    save_path=save_path,
                    class_name=class_names[i],
                    do_show=do_show,
                )
        else: # Draw a heatmap/segmentation map containing all classes in the dataset
            if save_folder is not None:
                save_path = os.path.join(save_folder, f"output_segmentations_img{image_idx}.{file_extension}")
            else:
                save_path = ""

            joint_prediction_segmentation_map(
                prediction,
                orig_image=img,
                class_names=class_names,
                title="({}) {}".format(image_idx + 1, input_image_path),
                save_path=save_path,
                do_show=do_show,
            )


def joint_prediction_segmentation_map(prediction, orig_image, class_names, *, title="", save_path="", do_show=False):
    prediction_map_2d = np.argmax(prediction, axis=-1)

    fig = display.display_segmentation(
        segmentation_map=prediction_map_2d,
        image=orig_image,  # input_image,
        class_name_list=class_names,
        title=title,
        save_to=save_path,
        do_show=False,
    )

    if do_show:
        plt.show()

    plt.close("all")


def single_channel_heatmap(prediction, orig_image, class_name, *, title="", save_path="", do_show=False):
    prediction = prediction * 255.0
    prediction = np.stack([prediction, np.zeros_like(prediction), np.zeros_like(prediction)], axis=-1)

    # Merge prediction with original image
    orig_image = orig_image.convert("L")

    prediction_img = Image.fromarray(prediction.astype("uint8"))
    # prediction_img = prediction_img.resize(img.size)

    img_ary = np.array(orig_image, dtype="uint8")
    pred_ary = np.array(prediction_img, dtype="uint8")

    if "inline" in matplotlib.get_backend():
        print(class_name)

    plt.figure(figsize=(14, 7))

    plt.imshow(pred_ary, alpha=1) # , vmax=len(class_names) - 1
    display.display_image(img_ary, title=title, do_show=do_show, alpha=0.75, new_fig=False, cmap="gray")

    plt.tight_layout()
    plt.axis("off")

    if save_path is not None:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(save_path,
                    bbox_inches="tight", pad_inches=0)

    plt.close()