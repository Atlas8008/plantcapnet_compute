import torch


def max_scale(x, dim=1, keepdim=True):
    ret = x / torch.amax(x, dim=dim, keepdim=keepdim)

    return torch.nan_to_num(ret, nan=0.0)

def threshold(x, thresh=0.5):
    return torch.where(
        x >= thresh,
        1.0,
        0.0,
    )

def apply_discretization(segmentations, deoc_segmentations, d_type, input_img, prediction_fun):
    """
    Apply discretization to the segmentations.

    Args:
        segmentations (torch.Tensor): The segmentations to discretize.
        deoc_segmentations (torch.Tensor): The deoc segmentations to combine with. Can be None.
        d_type (str): The discretization type. Can be one of:
            - "argmax" - Select the class with the highest score.
            - "scaled" - Scale the segmentations to [0, 1] using the maximum value.
            - "scaled_thresh" - Scale the segmentations to [0, 1] and apply a threshold at 0.5.
            - "scaled_thresh_deoc" - Scale the segmentations to [0, 1], apply a threshold at 0.5, and combine with deoc segmentations. Deprecated.
            - "combined" - Combine the segmentations with deoc segmentations. Without deoc segmentations, this is equivalent to "argmax".
            - "self_looped" - Apply self-looped segmentation using deocclusion algorithms. Deprecated.
            - "self_looped_scale" - Apply self-looped segmentation using deocclusion algorithms and scale the result. Deprecated.
            - "none" or None - No discretization.
        input_img (torch.Tensor): The input image.
        prediction_fun (callable): The function to use for prediction.
    """
    assert d_type in ("argmax", "scaled", "scaled_thresh", "scaled_thresh_deoc", "combined", "self_looped", "self_looped_scale", "none", None)

    if d_type in ("none", None):
        return segmentations

    if d_type in ("argmax", "combined", "self_looped"):
        segmentations = torch.where(
            segmentations == torch.amax(segmentations, dim=1, keepdim=True),
            1.0,
            0.0,
        )
    elif d_type in ("scaled", "scaled_thresh", "self_looped_scale"):
        segmentations = max_scale(segmentations)

        if d_type != "scaled":
            segmentations = threshold(segmentations)

    if d_type in ("combined", "scaled_thresh_deoc"):
        if deoc_segmentations is not None:
            deoc_segmentations = deoc_segmentations / torch.amax(deoc_segmentations, dim=1, keepdim=True)

            deoc_segmentations = threshold(deoc_segmentations)

            # Combine segmentations with deoc segmentations for calculation
            if d_type == "combined":
                combination = torch.maximum(segmentations[:, :-1], deoc_segmentations)
            elif d_type == "scaled_thresh_deoc":
                combination = deoc_segmentations
            else:
                raise ValueError()

            segmentations = torch.cat([
                combination,
                segmentations[:, -1:],
            ], dim=1)
        elif d_type == "scaled_thresh_deoc":
            raise ValueError("Invalid discretization method for current network setup.")
    elif d_type in ("self_looped", "self_looped_scale"):
        deoc_segmentations = []

        for class_idx in range(segmentations.shape[1] - 1): # Skip background
            class_masked_input = input_img * segmentations[:, class_idx:class_idx + 1]

            deoc_segmentation = prediction_fun(class_masked_input)

            deoc_segmentations.append(deoc_segmentation[:, class_idx])

        deoc_segmentations = torch.sigmoid(torch.stack(deoc_segmentations, dim=1))

        deoc_segmentations = max_scale(deoc_segmentations)
        deoc_segmentations = threshold(deoc_segmentations)

        # Add bg again
        deoc_segmentations = torch.cat([deoc_segmentations, segmentations[:, -1:]], dim=1)

        segmentations = torch.maximum(segmentations, deoc_segmentations)

    return segmentations