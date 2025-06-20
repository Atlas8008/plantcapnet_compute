import os
import gc
import torch
import numpy as np

from torch import nn
from PIL import Image
from imgaug import augmenters
from torch.utils.data import DataLoader

from utils.postprocessing import predict_enriched_output_by_augmentation
from utils.segmentation_utils import index_map_to_image
from utils.utils import LineUpdater


def segmentation_resize_augmentations(image_size):
    """The default augmentations for WSOL inference.
    This is used to resize the input images to a specific size and
    then crop them to multiples of 32.
    Args:
        image_size (int): The target size for resizing the images.
    Returns:
        list: A list of augmentations to be applied to the images.
    """
    return [
        np.asarray,
        augmenters.Sequential([
                augmenters.Resize(
                    {"shorter-side": int(image_size / 0.875),
                     "longer-side": "keep-aspect-ratio"}
                ),
                augmenters.CenterCropToMultiplesOf(height_multiple=32, width_multiple=32)
        ]).augment_image,
        Image.fromarray,
    ]

def discretize(pred, threshold):
    try:
        pred = torch.where(
            pred > threshold * torch.amax(pred),
            1.0,
            0.0,
        )
    except torch.cuda.OutOfMemoryError:
        # Fallback: process everything on CPU
        device = pred.device
        pred = pred.cpu()

        gc.collect()
        torch.cuda.empty_cache()

        pred = torch.where(
            pred > threshold * torch.amax(pred),
            1.0,
            0.0,
        )

        #pred = pred.to(device)

    return pred


def _mem():
    print(f"torch.cuda.memory_allocated: {(torch.cuda.memory_allocated(0)/1024/1024/1024)}GB")
    print(f"torch.cuda.memory_reserved: {(torch.cuda.memory_reserved(0)/1024/1024/1024)}GB")
    print(f"torch.cuda.max_memory_reserved: {(torch.cuda.max_memory_reserved(0)/1024/1024/1024)}GB")


def output_cam_segmentations(model, dataset, target_folder, cam_threshold=0.2, device=None, enriched_output=False, model_mode="cam", resize_to_original=False, save_path_keep_folder_levels=1, check_existing=True, **model_kwargs):
    """
    Perform CAM segmentation on a dataset and save the results to a specified folder. This function uses a model to generate segmentations based on Class Activation Mapping (CAM) and saves the output images.

    Args:
        model (torch.nn.Module): The model to use for segmentation.
        dataset (torch.utils.data.Dataset): The dataset to perform inference on.
        target_folder (str): The folder where the output segmentations will be saved.
        cam_threshold (float): The threshold for CAM segmentation.
        device (torch.device): The device to perform inference on.
        enriched_output (bool): Flag indicating whether to use enriched WSOL output (with test time augmentation).
        model_mode (str): The mode for the model (e.g., "cam").
        resize_to_original (bool): Flag indicating whether to resize the output to the original image size.
        save_path_keep_folder_levels (int): Number of folder levels to keep in the output path.
        **model_kwargs: Additional keyword arguments for the model.
    """
    if model_mode is not None and len(model_kwargs) == 0:
        model_kwargs["mode"] = model_mode

    model.eval()

    class_names = dataset.class_names

    has_one_hot_labels = hasattr(dataset, "has_one_hot_labels") and dataset.has_one_hot_labels
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    #colormap = np.stack([np.array([1, 1, 1]) * i for i in range(len(class_names) + 1)], axis=0)
    colormap = np.stack([(np.array([1, 1, 1]) * i + [i // 256, 0, 0]) % 256 for i in range(len(class_names) + 1)], axis=0) # Added shift to support more than 256 classes

    pred_list = tuple(zip(*dataset.data_list))[1]

    with LineUpdater() as lu:
        for idx, (img_name, (x, y)) in enumerate(zip(pred_list, data_loader)):
            lu.print(f"Image {idx + 1}/{len(data_loader)}")

            sub_folders = os.path.join(
                *os.path.normpath(img_name).split(os.path.sep)[-1 - save_path_keep_folder_levels:-1])


            segmentation_fpath = os.path.join(
                target_folder,
                sub_folders,
                os.path.splitext(os.path.basename(img_name))[0] + ".png"
            )

            if check_existing and os.path.exists(segmentation_fpath):
                continue

            gc.collect()
            torch.cuda.empty_cache()

            if device is not None:
                x = x.to(device)
                y = y.to(device)

            with torch.amp.autocast(device.type), torch.no_grad():
                # Filter out all labels not belonging to class
                if not has_one_hot_labels:
                    one_hot_labels = nn.functional.one_hot(y[0], len(class_names))[None, :, None, None]
                else:
                    one_hot_labels = y[0][None, :, None, None]

                if not enriched_output:
                    pred = model(x, **model_kwargs)
                else:
                    pred = predict_enriched_output_by_augmentation(
                        x,
                        model=model,
                        device=device,
                        scales=(1,),
                        vertical_flips=True,
                        horizontal_flips=True,
                        interlaced=False,
                        resize_inputs_to_multiples_of=32,
                        model_kwargs=model_kwargs,
                    )

                if resize_to_original:
                    w, h = Image.open(img_name).size
                else:
                    h, w = x.shape[2], x.shape[3]

                try:
                    pred = nn.functional.interpolate(pred, (h, w), mode="bilinear")
                    pred *= one_hot_labels
                except torch.cuda.OutOfMemoryError:
                    pred = nn.functional.interpolate(pred.cpu(), (h, w), mode="bilinear")
                    pred *= one_hot_labels.cpu()

                # Remove nonexistent classes from prediction

                del x, one_hot_labels

                gc.collect()
                torch.cuda.empty_cache()

                pred = discretize(pred, cam_threshold)

                bg = torch.where(
                    torch.sum(pred, dim=1) == 0,
                    1.0,
                    0.0,
                ).reshape((1, 1, pred.shape[2], pred.shape[3]))

                segmentation = torch.cat([bg, pred], dim=1)
                segmentation = torch.moveaxis(segmentation, 1, -1)

                index_map = torch.argmax(segmentation, dim=-1)

                index_map = index_map.cpu().numpy()[0]

                del pred, bg, segmentation

            segmentation_image = index_map_to_image(
                index_map,
                colors=colormap,
            )

            os.makedirs(os.path.split(segmentation_fpath)[0], exist_ok=True)

            segmentation_image.save(segmentation_fpath)
