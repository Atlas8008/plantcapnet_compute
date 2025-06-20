import torch
import torchmetrics

from torch import Tensor
from typing import Optional
from torchmetrics.utilities.distributed import reduce
from torchmetrics import JaccardIndex

from utils.torch_utils import rescale


def print_call(fn):
    def newfn(*args, **kwargs):
        print("\n" * 5 + "Call", fn.__name__)
        ret = fn(*args, **kwargs)
        print("\n" * 5 + "End of call", fn.__name__)
        return ret
    return newfn


class LabelIgnoreIoU(JaccardIndex):
    def __init__(self, *args, gt_map_scale_factor=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gt_map_scale_factor = gt_map_scale_factor

    def update(self, preds, target) -> None:
        with torch.no_grad():
            preds = torch.argmax(preds.detach(), dim=1)

            if self.gt_map_scale_factor:
                target = rescale(target, self.gt_map_scale_factor)

                return super().update(preds, target)

    def compute(self) -> Tensor:
        """
        Computes intersection over union (IoU)
        """
        return self._iou_from_confmat(self.confmat, self.num_classes, self.ignore_index, self.absent_score, self.reduction)

    def _iou_from_confmat(
        self,
        confmat: Tensor,
        num_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'elementwise_mean',
    ) -> Tensor:
        intersection = torch.diag(confmat)
        union = confmat.sum(0) + confmat.sum(1) - intersection

        # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
        scores = intersection.float() / union.float()
        scores[union == 0] = absent_score

        # Remove the ignored class index from the scores.
        if ignore_index is not None and 0 <= ignore_index < num_classes:
            scores = torch.cat([
                scores[:ignore_index],
                scores[ignore_index + 1:],
            ])
        return reduce(scores, reduction=reduction)


class ContinuousDiceScore(torchmetrics.Metric):
    _expensive = True
    def __init__(
        self,
        apply_sigmoid_to_preds=False,
        apply_sigmoid_to_targets=False,
        binary_classification=False,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn)

        self.apply_sigmoid_to_preds = apply_sigmoid_to_preds
        self.apply_sigmoid_to_targets = apply_sigmoid_to_targets
        self.binary_classification = binary_classification


        self.add_state("numerator", default=torch.zeros((1,)), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.zeros((1,)), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.to(torch.float32)
        target = target.to(torch.float32)

        if self.apply_sigmoid_to_preds:
            preds = torch.sigmoid(preds)

        if self.apply_sigmoid_to_targets:
            target = torch.sigmoid(target)

        if self.binary_classification:
            self.numerator += 2.0 * (torch.sum(preds * target) + torch.sum((1 - preds) * (1 - target)))
            self.denominator += torch.sum(preds) + torch.sum(target) + torch.sum(1 - preds) + torch.sum(1 - target)
        else:
            self.numerator += 2.0 * torch.sum(preds * target)
            self.denominator += torch.sum(preds) + torch.sum(target)

    def compute(self):
        return self.numerator / self.denominator
