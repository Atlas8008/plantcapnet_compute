import torch
import numpy as np

import torch.nn.functional as F

from abc import ABC, abstractmethod


class WeightedLoss(torch.nn.Module, ABC):
    def __init__(self, weights=None) -> None:
        super().__init__()

        self.weights = weights

    def _apply_weightings(self, err, input, target):
        weightings = torch.zeros_like(err)

        weights_set = False

        if self.weights is None:
            vals, counts = torch.unique(target, return_counts=True)

            if len(vals) > 1:
                for val, count in zip(vals, counts):
                    weightings[target == val] = 1 / count

                # Scale up so that the largest weighted value has a weight of 1
                weightings = weightings / torch.max(weightings)

                weights_set = True
        else:
            for k, v in self.weights.items():
                if k != "other":
                    if isinstance(v, np.ndarray):
                        v = torch.tensor(v)

                    if isinstance(v, torch.Tensor):
                        v = v.to(weightings.get_device())

                        reps = [1 for _ in target.shape]
                        reps[0] = target.shape[0]

                        weightings[target == k] = torch.tile(
                            v[None], reps)[target == k]
                    else:
                        weightings[target == k] = v

            if "other" in self.weights.keys():
                vals = torch.unique(target)

                for val in vals:
                    if val.item() not in self.weights:
                        v = self.weights["other"]

                        if isinstance(v, np.ndarray):
                            v = torch.tensor(v)

                        if isinstance(v, torch.Tensor):
                            v = v.to(weightings.get_device())

                            reps = [1 for _ in target.shape]
                            reps[0] = target.shape[0]

                            weightings[target == k] = torch.tile(
                                v[None], reps)[target == k]
                        else:
                            weightings[target == k] = v

                weights_set = True

        if weights_set:
            err *= weightings

        return err


class WeightedMSELoss(WeightedLoss):
    def forward(self, input, target):
        err = (input - target) ** 2
        err = self._apply_weightings(err, input, target)

        return torch.mean(err)


class WeightedRMSELoss(WeightedLoss):
    def forward(self, input, target):
        err = (input - target) ** 2
        err = self._apply_weightings(err, input, target)

        err = torch.sqrt(
            torch.mean(
                err,
                dim=tuple(i for i in range(1, len(err.shape)))
            )
        )

        return torch.mean(err)


class WeightedL1Loss(WeightedLoss):
    def forward(self, input, target):
        err = torch.abs(input - target)
        err = self._apply_weightings(err, input, target)

        return torch.mean(err)


class RMSELoss(WeightedLoss):
    def forward(self, input, target):
        err = (input - target) ** 2

        err = torch.sqrt(
            torch.mean(
                err,
                dim=tuple(i for i in range(1, len(err.shape)))
            )
        )

        return torch.mean(err)


class MPNELoss(WeightedLoss):
    def __init__(self, n, weights=None) -> None:
        super().__init__(weights)

        self.n = n

    def forward(self, input, target):
        err = torch.abs(input - target) ** self.n

        return torch.mean(err)


class MeanScaledAbsoluteErrorLoss(torch.nn.Module):
    def __init__(self, means, stds, class_mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(means) == len(stds)

        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means)
        if not isinstance(stds, torch.Tensor):
            stds = torch.tensor(stds)

        self.means = means
        self.stds = stds

        self.class_mask = torch.tensor(class_mask, dtype=torch.bool) if class_mask else None

    def forward(self, preds, targets):
        means = self.means.to(preds.device)[None]
        stds = self.stds.to(preds.device)[None]

        if len(preds.shape) > len(means.shape):
            means = means[:, :, None]
            stds = stds[:, :, None]

        preds = (preds.to(torch.float32) - means) / stds
        targets = (targets.to(torch.float32) - means) / stds

        if self.class_mask:
            preds = preds[:, self.class_mask]
            targets = targets[:, self.class_mask]

        errors = torch.mean(torch.abs(preds - targets))

        return errors


class MeanSquaredNormalizedErrorLoss(MeanScaledAbsoluteErrorLoss):
    def forward(self, preds, targets):
        means = self.means.to(preds.device)[None]
        stds = self.stds.to(preds.device)[None]

        if len(preds.shape) > len(means.shape):
            means = means[:, :, None]
            stds = stds[:, :, None]

        preds = (preds.to(torch.float32) - means) / stds
        targets = (targets.to(torch.float32) - means) / stds

        if self.class_mask:
            preds = preds[:, self.class_mask]
            targets = targets[:, self.class_mask]

        errors = torch.mean((preds - targets) ** 2)

        return errors



class ExponentialAbsoluteErrorLoss(torch.nn.Module):
    def forward(self, preds, targets):
        errors = torch.mean(torch.exp(1 - targets) * torch.abs(preds - targets))

        return errors


class BhattacharyyaLoss(torch.nn.Module):
    def __init__(self, smoothing_value=0.01, epsilon=0, coefficient_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.smoothing_value = smoothing_value
        self.epsilon = epsilon
        self.coefficient_only = coefficient_only

    def forward(self, preds, targets):
        preds = torch.clamp(preds, 0.0, 1.0) + self.smoothing_value
        targets = torch.clamp(targets, 0.0, 1.0) + self.smoothing_value

        prod = torch.clamp(preds * targets, 0.0, 1.0)

        bc = torch.sum(torch.sqrt(prod) - self.smoothing_value)

        if self.coefficient_only:
            return bc

        return -torch.log(bc + self.epsilon)

class HellingerLoss(torch.nn.Module):
    def forward(self, preds, targets):
        preds = torch.clamp(preds, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        return 1 / 2 ** 0.5 * torch.sqrt(torch.sum((torch.sqrt(preds) - torch.sqrt(targets)) ** 2))


class ShannonIndexLoss(torch.nn.Module, ABC):
    @staticmethod
    def _shannon_index(x):
        return -torch.nansum(x * torch.log(x))

    @abstractmethod
    def _difference_fun(self, preds, targets):
        return None

    def forward(self, preds, targets):
        preds = self._shannon_index(preds)
        targets = self._shannon_index(targets)

        return torch.mean(self._difference_fun(preds, targets))

class ShannonIndexMAELoss(ShannonIndexLoss):
    def _difference_fun(self, preds, targets):
        return torch.abs(preds - targets)

class ShannonIndexMSELoss(ShannonIndexLoss):
    def _difference_fun(self, preds, targets):
        return (preds - targets) ** 2

class ShannonIndexRMSELoss(ShannonIndexLoss):
    def _difference_fun(self, preds, targets):
        return (preds - targets) ** 2

class KLDLoss(torch.nn.Module):
    def __init__(self, reduction='mean', smoothing_value=0.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.smoothing_value = smoothing_value

    def forward(self, input, target):
        # Transform input into log space
        input = torch.log(input + self.smoothing_value)

        return F.kl_div(input, target + self.smoothing_value, reduction=self.reduction, log_target=False)

class JensenShannonDivergence(torch.nn.Module):
    def __init__(self, reduction="mean", smoothing_value=0.0):
        super().__init__()
        self.reduction = reduction
        self.kld = KLDLoss(reduction="batchmean", smoothing_value=smoothing_value)

    def forward(self, input, target):
        m = 0.5 * (input + target)

        return 0.5 * self.kld(input, m) + 0.5 * self.kld(target, m)


class PCBCE(torch.nn.BCELoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.clamp = (0, 1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.clamp(input, self.clamp[0], self.clamp[1])

        return super().forward(input, target)