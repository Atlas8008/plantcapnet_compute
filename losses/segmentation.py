import torch

from torch import nn

from utils.torch_utils import rescale


class DiceLoss(torch.nn.Module):
    def __init__(self, apply_sigmoid_to_preds=False, apply_sigmoid_to_targets=False, index_map_targets=False, gt_map_scale_factor=None, ignore_index=None, smoothing_value=1.0,
        binary_classification=False):
        super().__init__()

        self.apply_sigmoid_to_preds = apply_sigmoid_to_preds
        self.apply_sigmoid_to_targets = apply_sigmoid_to_targets
        self.smoothing_value = smoothing_value
        self.index_map_targets = index_map_targets
        self.ignore_index = ignore_index
        self.gt_map_scale_factor = gt_map_scale_factor
        self.binary_classification = binary_classification

    def forward(self, preds, targets, mask=None):
        preds = preds.to(torch.float32)

        if self.gt_map_scale_factor:
            targets = rescale(targets, self.gt_map_scale_factor)

        if self.index_map_targets:
            if self.ignore_index:
                targets = torch.nn.functional.one_hot(targets, preds.shape[1] + 1)

                # Remove index to be ignored
                targets = torch.cat([
                        targets[..., :self.ignore_index],
                        targets[..., self.ignore_index + 1:]
                    ],
                    dim=-1,
                )
            else:
                targets = torch.nn.functional.one_hot(targets, preds.shape[1])

            targets = torch.moveaxis(targets, -1, 1)
        else:
            targets = targets.to(torch.float32)

        if self.apply_sigmoid_to_preds:
            preds = torch.sigmoid(preds)
        if self.apply_sigmoid_to_targets:
            targets = torch.sigmoid(targets)

        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:, None]

            preds = preds * mask
            targets = targets * mask

        if self.binary_classification:
            numerator = 2.0 * (torch.sum(preds * targets) + torch.sum((1 - preds) * (1 - targets))) + self.smoothing_value
            denominator = torch.sum(preds) + torch.sum(targets) + torch.sum(1 - preds) + torch.sum(1 - targets) + self.smoothing_value
        else:
            numerator = 2.0 * torch.sum(preds * targets) + self.smoothing_value
            denominator = torch.sum(preds) + torch.sum(targets) + self.smoothing_value

        return 1.0 - numerator / denominator


class ClassFocusedBCEDice(torch.nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super().__init__()

        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

        self.bce_loss = nn.BCELoss(reduction="none")
        self.dice_loss = DiceLoss()

    def forward(self, preds, targets):
        contains_class = torch.sum(targets, dim=(2, 3)) > 0
        contains_class_float = contains_class.to(torch.float32)

        bce_mean = self.bce_loss(preds, targets)
        bce_mean = torch.where(
            contains_class,
            torch.zeros_like(contains_class_float),
            torch.mean(bce_mean, dim=(2, 3)),
        )
        bce_mean = torch.sum(bce_mean, dim=1) / torch.sum(1 - contains_class_float, dim=1)
        bce = torch.mean(bce_mean)

        dice = self.dice_loss(preds, targets)

        return self.weight_bce * bce + self.weight_dice * dice
