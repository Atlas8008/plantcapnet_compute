import torch
import torchmetrics


def _maybe_expand(tensor, shape):
    if len(tensor.shape) < len(shape):
        return tensor[..., None]
    else:
        return tensor


class MeanAbsoluteErrorWClassMask(torchmetrics.MeanAbsoluteError):
    def __init__(self, class_mask, **kwargs) -> None:
        super().__init__(**kwargs)

        self.class_mask = torch.tensor(class_mask, dtype=torch.bool)

    def update(self, preds, target) -> None:
        if self.class_mask is not None:
            preds = preds[:, self.class_mask]
            target = target[:, self.class_mask]

        return super().update(preds, target)


class MeanScaledAbsoluteError(torchmetrics.Metric):
    def __init__(self, scale_values, classwise=False, class_mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if classwise:
        #     self.add_state("errors", default=torch.tensor((0.0,)), dist_reduce_fx="sum")
        #     self.add_state("n_elems", default=torch.tensor((0.0,)), dist_reduce_fx="sum")
        # else:
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_elems", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scale_values", default=torch.Tensor(scale_values), dist_reduce_fx="sum")

        self.class_mask = torch.tensor(class_mask, dtype=torch.bool) if class_mask else None

        self.classwise = classwise

    def update(self, preds, target):
        scale_values = _maybe_expand(self.scale_values[None], preds.shape)

        classwise_errors = torch.abs(preds - target) / scale_values

        if self.class_mask is not None:
            classwise_errors = classwise_errors[:, self.class_mask]

        if not self.classwise:
            self.errors += torch.sum(classwise_errors)

            self.n_elems += torch.numel(preds)
        else:
            self.errors = torch.sum(classwise_errors, dim=0) + self.errors

            self.n_elems += preds.shape[0]

    def compute(self):
        return self.errors / self.n_elems


class MeanNormalizedAbsoluteError(torchmetrics.Metric):
    def __init__(self, means, stds, class_mask=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(means) == len(stds)

        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means)
        if not isinstance(stds, torch.Tensor):
            stds = torch.tensor(stds)

        self.means = means
        self.stds = stds

        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_elems", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.class_mask = torch.tensor(class_mask, dtype=torch.bool) if class_mask else None

    def update(self, preds, target):
        self.means = self.means.to(preds.device)
        self.stds = self.stds.to(preds.device)

        means = _maybe_expand(self.means[None], preds.shape)
        stds = _maybe_expand(self.stds[None], preds.shape)

        preds = (preds - means) / stds
        target = (target - means) / stds

        if self.class_mask is not None:
            preds = preds[:, self.class_mask]
            target = target[:, self.class_mask]

        self.errors = torch.sum(torch.abs(preds - target), dim=0) + self.errors
        self.n_elems += preds.shape[0]

    def compute(self):
        return torch.mean(self.errors / self.n_elems)


# class MeanSelfNormalizedAbsoluteError(torchmetrics.Metric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.add_state("errors", default=list, dist_reduce_fx="cat")

#     def update(self, preds, target):
#         for tensor in torch.abs(preds - target):
#             self.errors.append(tensor)

#     def compute(self):
#         final_tensor = torch.stack(self.errors)

#         stds, means = torch.std_mean(final_tensor, dim=0, unbiased=False)

#         return torch.mean(self.errors / self.n_elems)

