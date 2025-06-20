import numpy as np

from scipy.interpolate import RectBivariateSpline


def resize_interpolate(ary, target_size, resample="BICUBIC"):
    resample_methods = {
        "NEAREST": 0,
        "BILINEAR": 1,
        "BICUBIC": 3,
    }

    resample = resample.upper()

    assert resample in resample_methods

    single_channels = []

    for i in range(ary.shape[-1]):
        single_channel = ary[..., i]

        spline = RectBivariateSpline(
            x=np.arange(ary.shape[0]),
            y=np.arange(ary.shape[1]),
            z=single_channel,
            kx=resample_methods[resample],
            ky=resample_methods[resample],
        )

        new_coords_x = np.arange(target_size[0]) * (single_channel.shape[0] / target_size[0])
        new_coords_y = np.arange(target_size[1]) * (single_channel.shape[1] / target_size[1])

        single_channel_interpolated = spline(new_coords_x, new_coords_y)

        single_channels.append(single_channel_interpolated)

    interpolated = np.stack(single_channels, axis=-1)

    return interpolated


def to_one_hot(a, nb_classes):
    new_shape = a.shape + (nb_classes,)

    one_hot = np.zeros(new_shape)

    expansion = tuple(range(len(new_shape) - 1))

    one_hot[
        a[..., None] ==
        np.expand_dims(np.arange(nb_classes), axis=expansion)
    ] = 1

    return one_hot


def bin_count(values, bin_min, bin_max, bin_step):
    bins = np.arange(bin_min, bin_max, step=bin_step)

    values = values.flatten()

    binned_idx = np.digitize(values, bins)
    counts = np.bincount(binned_idx, minlength=len(bins))
    
    return bins, counts
