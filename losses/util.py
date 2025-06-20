import torch
import torch.nn.functional as F

from torch import nn
from functools import wraps


def cast_method_inputs(obj, fun_name, preds_type=None, target_type=None):
    fn = getattr(obj, fun_name)

    @wraps(fn)
    def new_fn(preds, target):
        if preds_type is not None:
            preds = preds.to(preds_type)
        if target_type is not None:
            target = target.to(target_type)

        return fn(preds, target)

    setattr(obj, fun_name, new_fn)

    return obj

def argmax_onehot(obj, fun_name, *, onehot_preds, onehot_target):
    fn = getattr(obj, fun_name)

    @wraps(fn)
    def new_fn(preds, target):
        if onehot_preds:
            n_classes = preds.shape[1]

            preds = torch.argmax(preds, 1)
            preds = F.one_hot(preds, n_classes)
            preds = torch.moveaxis(preds, -1, 1)
        if onehot_target:
            n_classes = target.shape[1]

            target = torch.argmax(target, 1)
            target = F.one_hot(target, n_classes)
            target = torch.moveaxis(target, -1, 1)

        return fn(preds, target)

    setattr(obj, fun_name, new_fn)

    return obj


def argmax(obj, fun_name, *, argmax_preds, argmax_target):
    fn = getattr(obj, fun_name)

    @wraps(fn)
    def new_fn(preds, target):
        if argmax_preds:
            preds = torch.argmax(preds, 1)
        if argmax_target:
            target = torch.argmax(target, 1)

        return fn(preds, target)

    setattr(obj, fun_name, new_fn)

    return obj



def squeeze_method_inputs(obj, fun_name, squeeze_preds=False, squeeze_target=False):
    fn = getattr(obj, fun_name)

    @wraps(fn)
    def new_fn(preds, target):
        print("p/t")
        print(preds.shape)
        print(target.shape)

        if squeeze_preds is not None:
            preds = torch.squeeze(preds)
        if squeeze_target is not None:
            target = torch.squeeze(target)

        return fn(preds, target)

    setattr(obj, fun_name, new_fn)

    return obj


class CombinedLoss(torch.nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = losses

        if weights is not None:
            assert len(weights) == len(losses), "The number of weights has to be equal to the number of losses provided."
        else:
            weights = [1.0] * len(losses)

        self.weights = [torch.tensor(w) for w in weights]


    def forward(self, preds, targets, mask=None):
        aggregate = torch.tensor(0.0).to(preds.device)

        if mask is not None:
            kwargs = {"mask": mask}
        else:
            kwargs = {}

        for weight, loss in zip(self.weights, self.losses):
            aggregate += weight.to(preds.device) * loss(preds, targets, **kwargs)

        return aggregate