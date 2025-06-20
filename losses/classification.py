import torch
import torch.nn.functional as F

from torch import nn


class BCEWithScalarsAndLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, n_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_classes = n_classes

    def forward(self, input, target):
        if self.n_classes is None:
            n_classes = input.shape[1]
        else:
            n_classes = self.n_classes

        target = nn.functional.one_hot(target, num_classes=n_classes).to(torch.float32)
        target = torch.moveaxis(target, -1, 1)

        return super().forward(input, target)


class BCEFocalLoss(nn.BCELoss):
    def __init__(self, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def forward(self, input, target):
        bce_unreduced = F.binary_cross_entropy(input, target, weight=self.weight)

        fl_unreduced = (1 - (input * target + (1 - input) * (1 - target))) ** self.gamma * bce_unreduced

        if self.reduction == "none":
            fl = fl_unreduced
        elif self.reduction == "mean":
            fl = torch.mean(fl_unreduced)
        elif self.reduction == "sum":
            fl = torch.sum(fl_unreduced)
        else:
            raise ValueError("Invalid reduction method: " + self.reduction)

        return fl