import torch
import numpy as np

from math import floor, ceil
from PIL import Image
from torchvision import transforms


def _randint(vmin, vmax, backend="numpy"):
    assert backend in ("numpy", "torch")

    if backend == "numpy":
        return np.random.randint(vmin, vmax)
    elif backend == "torch":
        return torch.randint(vmin, vmax, (1,))
    else:
        raise ValueError("Invalid backend: " + backend)

def _indices(shape, backend="numpy", device=None):
    assert backend in ("numpy", "torch")

    if backend == "numpy":
        indices = np.indices(shape)
    elif backend == "torch":
        indices = torch.stack(
            torch.meshgrid(
                [torch.arange(s, device=device) for s in shape],
                indexing="ij"
            )
        )

    return indices


def _sample_mask_size(min_mask_size, max_mask_size, independent_wh):

    mask_size_x = _randint(min_mask_size, max_mask_size + 1, backend="numpy")

    if independent_wh:
        mask_size_y = _randint(
            min_mask_size, max_mask_size + 1, backend="numpy")
    else:
        mask_size_y = mask_size_x

    return mask_size_x, mask_size_y

def _apply_cutout(ary, x, y, mask_size_x, mask_size_y, circular=False, backend="numpy"):
    assert backend in ("numpy", "torch")
    if backend == "numpy":
        h, w, _ = ary.shape
        ary = ary[None]
        device = None
    elif backend == "torch":
        _, h, w = ary.shape
        ary = ary[..., None]
        device = ary.get_device()
        if device == -1:
            device = None

    if not circular:
        ary[
            :,
            max(0, y - floor(mask_size_y / 2)):min(h - 1, y + ceil(mask_size_y / 2)),
            max(0, x - floor(mask_size_x / 2)):min(w - 1, x + ceil(mask_size_x / 2)),
            :,
        ] = 0
    else:
        # Cut out only relevant image parts to prevent distance calculations for every single pixel
        y_min = max(0, y - floor(mask_size_y / 2))
        y_max = min(h - 1, y + ceil(mask_size_y / 2))
        x_min = max(0, x - floor(mask_size_x / 2))
        x_max = min(w - 1, x + ceil(mask_size_x / 2))

        indices = _indices((h, w), backend, device=device)[:, y_min:y_max, x_min:x_max]

        # Center indices around x and y
        indices[0] -= y
        indices[1] -= x

        r_x = mask_size_x / 2
        r_y = mask_size_y / 2

        if backend == "numpy":
            radii = np.array([r_y, r_x])[:, None, None]
            dists = np.sum(indices ** 2 / radii ** 2, axis=0)

            ary[:, y_min:y_max, x_min:x_max, :][dists[None] < 1] = 0
        elif backend == "torch":
            radii = torch.tensor([r_y, r_x], device=device)[:, None, None]
            dists = torch.sum(indices ** 2 / radii ** 2, axis=0)

            dists = dists[None, ..., None]

            ary[:, y_min:y_max, x_min:x_max, :] *= torch.where(
                dists < 1,
                torch.zeros_like(dists, dtype=ary.dtype),
                torch.ones_like(dists, dtype=ary.dtype),
            )

    if backend == "numpy":
        ary = ary[0]
    elif backend == "torch":
        ary = ary[..., 0]

    return ary


def cutout(img, min_mask_size: int, max_mask_size: int, independent_wh=False, reps=1, border_contained=False, circular=False, backend="auto"):
    is_image = isinstance(img, Image.Image)

    if is_image:
        img = np.array(img)

    if backend == "auto":
        if isinstance(img, torch.Tensor):
            backend = "torch"
        else:
            backend = "numpy"

    if backend == "numpy":
        h, w, _ = img.shape
        valid = np.ones_like(img[..., 0:1])
    elif backend == "torch":
        _, h, w = img.shape
        valid = torch.ones_like(img[0:1])

    for i in range(reps):
        mask_size_x, mask_size_y = _sample_mask_size(
            min_mask_size,
            max_mask_size,
            independent_wh,
        )

        if not border_contained:
            x, y = np.random.randint(w), np.random.randint(h)
        else:
            half_mask_x = ceil(mask_size_x / 2)
            half_mask_y = ceil(mask_size_y / 2)

            x = np.random.randint(half_mask_x, w - half_mask_x)
            y = np.random.randint(half_mask_y, h - half_mask_y)

        valid = _apply_cutout(
            valid,
            x=x,
            y=y,
            mask_size_x=mask_size_x,
            mask_size_y=mask_size_y,
            circular=circular,
            backend=backend,
        )

    img = img * valid

    if is_image:
        return Image.fromarray(img)
    else:
        return img


def inverted_cutout(img, min_mask_size: int, max_mask_size: int, independent_wh=False, reps=1, border_contained=False, circular=False, backend="auto"):
    is_image = isinstance(img, Image.Image)

    if is_image:
        img = np.array(img)

    if backend == "auto":
        if isinstance(img, torch.Tensor):
            backend = "torch"
        else:
            backend = "numpy"

    if backend == "numpy":
        valid = np.ones_like(img[..., 0:1])
    elif backend == "torch":
        valid = torch.ones_like(img[0:1])

    valid = cutout(
        valid,
        min_mask_size,
        max_mask_size,
        independent_wh=independent_wh,
        reps=reps,
        border_contained=border_contained,
        circular=circular,
        backend=backend,
    )

    valid = 1 - valid

    img = img * valid

    if is_image:
        return Image.fromarray(img)
    else:
        return img


def generate_validation_occlusion_map(img_size, val_min, val_max, tile_size, gap_size, seed=1337):
    np.random.seed(seed)

    noise_map = np.random.random(img_size + (3, )) * (val_max + val_min) - val_min

    mask = np.ones((gap_size + tile_size, gap_size + tile_size))
    mask[gap_size:, gap_size:] = 0

    mask = np.tile(
        mask,
        (ceil(img_size[0] / (gap_size + tile_size)), ceil(img_size[1] / (gap_size + tile_size)))
    )[:img_size[0], :img_size[1]]

    return noise_map, mask


def validation_cutout(img, replacement_map, mask):
    is_image = isinstance(img, Image.Image)

    if is_image:
        img = np.array(img)

    img = np.where(
        mask[:, :, None],
        img,
        replacement_map,
    )

    if is_image:
        return Image.fromarray(img.astype("uint8"))
    else:
        return img
