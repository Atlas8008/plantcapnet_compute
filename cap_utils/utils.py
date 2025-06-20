import os
import json
import torch
import matplotlib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from torch import nn
from torchvision import transforms
from tqdm import tqdm
from functools import wraps
from multiprocessing import cpu_count

from utils.augmentations import torchaug as torchaug
from losses import DiceLoss, CombinedLoss, BCEFocalLoss, ClassFocusedBCEDice

from utils.wsol import output_cam_segmentations


def eval_and_save(evaluation_method, config, model=None, device=None):
    if model is not None:
        metric_results = evaluation_method(model, device)
    else:
        metric_results = evaluation_method()

    os.makedirs(
        config.metric_path,
        exist_ok=True,
    )

    with open(os.path.join(config.metric_path, evaluation_method.name + ".json"), "w") as f:
        f.write(json.dumps(metric_results, indent=True))

def get_arg_attribute(args, mode, attr):
    return getattr(args, mode + "_" + attr)

def get_num_workers(default):
    if "SLURM_CPUS_ON_NODE" in os.environ:
        min_cpus = min(cpu_count(), int(os.environ["SLURM_CPUS_ON_NODE"]))
    else:
        min_cpus = cpu_count()

    return min(min_cpus, default)

def check_model_parameter_counts(param_list, model):
    single_weights = sum(map(len, param_list))
    model_total_weights = len(list(model.parameters()))

    # Check, if all params are accounted for
    assert single_weights == model_total_weights, f"The number of weights in the submodules and total model parameters does not match. {single_weights} != {model_total_weights}"

def output_model_segmentations(model, target_folder, train_dataset, val_dataset, device, cam_threshold=0.0):

    output_cam_segmentations(
        model=model,
        dataset=train_dataset,
        target_folder=target_folder,
        device=device,
        cam_threshold=cam_threshold,
    )
    output_cam_segmentations(
        model=model,
        dataset=val_dataset,
        target_folder=target_folder,
        device=device,
        cam_threshold=cam_threshold,
    )


def get_loss(name):
    if name == "dice":
        loss = DiceLoss()  #nn.BCEWithLogitsLoss()
    elif name == "bce":
        loss = MaskedBCE()
    elif name == "bce_dice":
        loss = CombinedLoss([DiceLoss(), MaskedBCE()])
    elif name == "cfbce_dice":
        loss = ClassFocusedBCEDice()
    elif name == "focal":
        loss = BCEFocalLoss()
    elif name == "mse":
        loss = MaskedMSE()
    else:
        raise ValueError("Invalid loss function: " + name)

    return loss


class MaskedBCE(nn.BCELoss):
    def forward(self, input, target, mask=None):
        loss = torch.nn.functional.binary_cross_entropy(
            input, target, weight=self.weight, reduction="none")

        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:, None]

            input = input * mask
            target = target * mask

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class MaskedMSE(nn.MSELoss):
    def forward(self, input, target, mask=None):
        loss = torch.nn.functional.mse_loss(
            input, target, reduction="none")

        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:, None]

            input = input * mask
            target = target * mask

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


def plot_and_save_model_results(model, image_transform, save_path, device, images=None, split=None, heatmap_per_channel=False, discretize=True, output_index=0, species_list=None, **model_kwargs):
    """
    Generates and saves class maps for the given model and images.
    Args:
        model: The model to generate class maps for.
        image_transform: The transformation to apply to the input images.
        save_path: The path to save the generated class maps.
        device: The device to run the model on.
        images: The input images to generate class maps for. If None, uses the dataset's images.
        split: The split of the dataset (e.g., "train", "val", "test"). Only for designating the save path. Can be None.
        heatmap_per_channel: Whether to generate a heatmap for each channel instead of a single segmentation map of the topmost classes.
        discretize: Whether to discretize the output.
        output_index: The index of the output layer to use for generating class maps.
        species_list: A list of species names for labeling the class maps.
        **model_kwargs: Additional keyword arguments for the model.
    """
    from utils.class_maps import get_default_plant_species, generate_class_maps
    matplotlib.use("agg")

    if split is not None:
        actual_save_path = os.path.join(save_path, split)
    else:
        actual_save_path = save_path

    os.makedirs(actual_save_path, exist_ok=True)

    if species_list is not None:
        class_names = species_list + ["Background", "Irrelevance"]
    else:
        class_names = get_default_plant_species(with_dead_litter=False)

    generate_class_maps(
        model,
        input_image_transform=image_transform,
        input_images=images,
        class_names=class_names,
        device=device,
        heatmap_per_channel=heatmap_per_channel,
        discretize=discretize,
        output_index=output_index,
        do_show=False,
        save_folder=actual_save_path,
        **model_kwargs,
    )

def results_to_file(data, folder, split, dataset, prefix="", output_index=0, multivalue_index=None, avg_modes=None, include_targets=True, data_indices=None):
    headers = dataset.class_names
    id_headers = dataset.index_names

    _, target_data, ids = dataset.input_target_index_items

    if data_indices is not None:
        ids = data_indices

    if include_targets:
        pti_to_file(
            predictions=data,
            targets=target_data,
            indices=ids,
            headers=headers,
            id_headers=id_headers,
            folder=folder,
            file_prefix=split + prefix,
            output_index=output_index,
            multivalue_index=multivalue_index,
        )
    else:
        pi_to_file(
            predictions=data,
            indices=ids,
            headers=headers,
            id_headers=id_headers,
            folder=folder,
            file_prefix=split + prefix,
            multivalue_index=multivalue_index,
        )

    if avg_modes:
        for k, v in avg_modes.items():
            avg_prefix = k

            if include_targets:
                pti_to_file(
                    predictions=data,
                    targets=target_data,
                    indices=ids,
                    headers=headers,
                    id_headers=id_headers,
                    folder=folder,
                    file_prefix=split + prefix + "_" + avg_prefix,
                    output_index=output_index,
                    multivalue_index=multivalue_index,
                    avg_key=v,
                )
            else:
                pi_to_file(
                    predictions=data,
                    indices=ids,
                    headers=headers,
                    id_headers=id_headers,
                    folder=folder,
                    file_prefix=split + prefix + "_" + avg_prefix,
                    multivalue_index=multivalue_index,
                    avg_key=v,
                )

def average_gt_list_over_index_key(key, indices, id_headers, gt_list):
    index = pd.MultiIndex.from_arrays(np.array(indices).T, names=id_headers)

    # Get all level names until specified key
    assert key in index.names
    levels = index.names[:index.names.index(key)] + [key]

    gt_df = pd.DataFrame(np.array(gt_list), index=index)
    gt_df = gt_df.groupby(level=levels, sort=False).mean()

    assert np.all(gt_df.isin((0, 1))), "The mean of the gt booleans is neither 0 nor 1; invalid disambiguation detected"

    gt_df = gt_df.astype(bool)

    return gt_df.to_numpy().flatten().tolist()


def average_over_index_key(key, predictions, targets, indices, id_headers, multivalue_index=None):
    index = pd.MultiIndex.from_arrays(np.array(indices).T, names=id_headers)

    predictions_orig = np.array(predictions).squeeze()
    if targets:
        targets_orig = np.array(targets).squeeze()
    else:
        targets_orig = None

    if multivalue_index is None:
        if len(predictions_orig.shape) == 3:
            multivals = range(predictions_orig.shape[2])
        else:
            multivals = [None]
    else:
        multivals = [multivalue_index]

    preds_out = []
    targs_out = []

    for multival in multivals:
        if multival is not None:
            predictions = predictions_orig[..., multival]
        else:
            predictions = predictions_orig

        pred_df = pd.DataFrame(predictions, index=index)

        # Get all level names until specified key
        assert key in index.names
        levels = index.names[:index.names.index(key)] + [key]

        pred_df = pred_df.groupby(level=levels, sort=False).mean()

        predictions = pred_df.to_numpy()
        preds_out.append(predictions)

        if targets_orig is not None:
            if multival is not None:
                targets = targets_orig[..., multival]
            else:
                targets = targets_orig

            targ_df = pd.DataFrame(targets, index=index)
            targ_df = targ_df.groupby(level=levels, sort=False).mean()

            targets = targ_df.to_numpy()
            targs_out.append(targets)

    indices = pred_df.index.to_numpy()

    if len(preds_out) > 1:
        predictions = np.stack(preds_out, axis=-1)
    else:
        predictions = preds_out[0]

    if targs_out:
        if len(targs_out) > 1:
            targets = np.stack(targs_out, axis=-1)
        else:
            targets = targs_out[0]

    return predictions, targets, indices


def pti_to_file(predictions, targets, indices, headers, id_headers, folder, file_prefix, output_index=0, multivalue_index=None, avg_key=None):
    assert len(targets) == len(predictions) == len(indices), f"{len(predictions)}, {len(targets)}, {len(indices)}"
    #folder = os.path.join(folder, experiment_name, tag)

    os.makedirs(folder, exist_ok=True)

    targets = [target[output_index] if isinstance(target, (tuple, list)) else target for target in targets]

    pred_fname = os.path.join(folder, file_prefix + "_predictions.csv")
    targets_fname = os.path.join(folder, file_prefix + "_targets.csv")
    ids_fname = os.path.join(folder, file_prefix + "_indices.csv")

    assert len(targets) == len(predictions) and len(indices) == len(predictions), f"{len(targets)}, {len(predictions)}, {len(indices)}, {len(predictions)}"

    if avg_key is not None:
        predictions, targets, indices = average_over_index_key(
            avg_key,
            predictions=predictions,
            targets=targets,
            indices=indices,
            id_headers=id_headers,
            multivalue_index=multivalue_index,
        )

    pred_fcontents = []
    target_fcontents = []
    ids_fcontents = []

    for pred, target, id in zip(predictions, targets, indices):
        if isinstance(target, (tuple, list)):
            target = target[output_index]

        assert pred.shape == target.shape

        if multivalue_index is not None and len(pred.shape) == 2:
            pred = pred[..., multivalue_index]
            target = target[..., multivalue_index]

        pred_fcontents.append(",".join(map(str, pred)))
        target_fcontents.append(",".join(map(str, target)))
        ids_fcontents.append(",".join(map(str, id)))

    with open(pred_fname, "w") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(pred_fcontents))

    with open(targets_fname, "w") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(target_fcontents))

    with open(ids_fname, "w") as f:
        f.write(",".join(id_headers) + "\n")
        f.write("\n".join(ids_fcontents))


def pi_to_file(predictions, indices, headers, id_headers, folder, file_prefix, multivalue_index=None, avg_key=None):
    assert len(predictions) == len(indices), f"{len(predictions)}, {len(indices)}"

    os.makedirs(folder, exist_ok=True)

    pred_fname = os.path.join(folder, file_prefix + "_predictions.csv")
    ids_fname = os.path.join(folder, file_prefix + "_indices.csv")

    if avg_key is not None:
        predictions, _, indices = average_over_index_key(
            avg_key,
            predictions=predictions,
            targets=None,
            indices=indices,
            id_headers=id_headers,
            multivalue_index=multivalue_index,
        )

    pred_fcontents = []
    ids_fcontents = []

    for pred, id in zip(predictions, indices):
        if multivalue_index is not None and len(pred.shape) == 2:
            pred = pred[..., multivalue_index]

        pred_fcontents.append(",".join(map(str, pred)))
        ids_fcontents.append(",".join(map(str, id)))

    with open(pred_fname, "w") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(pred_fcontents))

    with open(ids_fname, "w") as f:
        f.write(",".join(id_headers) + "\n")
        f.write("\n".join(ids_fcontents))


def get_augmentations(augmentation_scheme, image_size, force_resize=False):
    """
    Returns the augmentation transforms for the given augmentation scheme.
    Args:
        augmentation_scheme (str): The augmentation scheme to use.
            - "rot": Applies random rotation to images and segmentations.
            - "rotrand": Applies random rotation and random augmentations to images and segmentations.
            - "rand": Applies random augmentations to images and segmentations.
            - "randrcrop": Applies random augmentations and random resized cropping to images and segmentations.
            - "rotrandrcrop": Applies random rotation, random augmentations, and random resized cropping to images and segmentations.
            - "rotrcrop": Applies random rotation and random resized cropping to images and segmentations.
            - "randblur": Applies random augmentations and Gaussian blur to images, while keeping segmentation blur in sync.
            - "randresize": Applies random augmentations and random resizing to images and segmentations.
        image_size (int): The size of the images.
        force_resize (bool): Whether to force resize the images to the given size.
    Returns:
        image_augs (list): The list of image augmentation transforms.
        segm_augs (list): The list of segmentation augmentation transforms.
    """
    image_augs = []
    segm_augs = []

    if augmentation_scheme == "rot":
        image_augs.append(transforms.RandomRotation((-180, 180), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomRotation(
            (-180, 180), interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "rotrand":
        image_augs.append(transforms.RandomRotation((-180, 180), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomRotation(
            (-180, 180), interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "rand":
        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "randrcrop":
        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "rotrandrcrop":
        image_augs.append(transforms.RandomRotation((-180, 180), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomRotation(
            (-180, 180), interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "rotrcrop":
        image_augs.append(transforms.RandomRotation((-180, 180), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomRotation(
            (-180, 180), interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(transforms.RandomResizedCrop(
            image_size, scale=(0.3, 1.0), interpolation=transforms.InterpolationMode.NEAREST))

    elif augmentation_scheme == "randblur":
        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

        # Segmentation blur should not change the segmentation, but still draw random numbers to keep RNG in sync with image augmenter
        segm_blur = input_return(transforms.GaussianBlur((5, 5)), "forward")

        image_augs.append(transforms.GaussianBlur((5, 5)))
        segm_augs.append(segm_blur)

    elif augmentation_scheme == "randresize":
        image_augs.append(transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR))
        segm_augs.append(RandAugmentSegmentation(interpolation=transforms.InterpolationMode.NEAREST))

        image_augs.extend([
                torchaug.RandomResize((0.1, 1), interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]
        )
        segm_augs.extend([
                torchaug.RandomResize((0.1, 1), interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST,
                )
            ]
        )
    else:
        raise ValueError("Invalid augmentation scheme: " + augmentation_scheme)

    if force_resize:
        image_augs.append(
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )
        segm_augs.append(
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )


    return image_augs, segm_augs


def input_return(obj, fun_name):
    fn = getattr(obj, fun_name)

    @wraps(fn)
    def new_fn(*args):
        fn(*args)

        if len(args) == 1:
            args = args[0]

        return args

    setattr(obj, fun_name, new_fn)

    return obj


class RandAugmentSegmentation(transforms.RandAugment):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0

            if op_name not in ("Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize", "AutoContrast", "Equalize"):
                img = transforms.autoaugment._apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img
