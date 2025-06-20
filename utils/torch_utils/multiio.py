import torch

from torch.nn import functional as F

__all__ = ["_to_list", "_is_list", "_all_equal", "_assert_compatible", "_all_to", "_unpack_tuple", "_predict", "_rescale_all", "_unscale_all"]

def _to_list(v):
    if v is None:
        return v
    if isinstance(v, dict): # Dicts are also valid multi inputs
        return v
    elif isinstance(v, tuple):
        return list(v)
    elif isinstance(v, list):
        return v
    else:
        return [v]

def _is_list(v):
    return isinstance(v, (list, tuple))

def _all_equal(l):
    return all(item == l[0] for item in l)

def _assert_compatible(*items):
    assert all(
        not _is_list(item) for item in items
    ) or _all_equal([len(item) for item in items]), f"Dimensions or item counts do not match. ({[len(item) for item in items]})"

def _all_to(l, device):
    if l is None:
        return l
    if isinstance(l, torch.Tensor):
        return l.to(device)
    if isinstance(l, dict):
        l = {k: _all_to(v, device) for k, v in l.items()}
    else:
        l = [_all_to(item, device) for item in l]

    return l

def _rescale_all(l, scale):
    return [
        F.interpolate(it, scale_factor=scale, mode="bilinear")
            for it in l
    ]

def _unscale_all(l, scale):
    return [
        F.interpolate(it, scale_factor=1 / scale, mode="bilinear")
            for it in l
    ]

def _unpack_tuple(t, device):
    if not isinstance(t, (tuple, list)):
        x = t
        y = None
        mask = None
    elif len(t) == 1:
        x = t[0]
        y = None
        mask = None
    elif len(t) == 2:
        x, y = t
        mask = None
    else:
        x, y, mask = t
        mask = mask.to(device)

    x = _to_list(x)
    y = _to_list(y)

    x, y = _all_to(x, device), _all_to(y, device)

    return x, y, mask

def _predict(x, model, autocast=True, **model_kwargs):
    x_ = x
    device = None

    # Get device for autocast
    while not isinstance(x_, torch.Tensor):
        if isinstance(x_, (tuple, list)):
            x_ = x[0]
        else:
            x_ = list(x.values())[0]

    device = x_.device

    with torch.amp.autocast(device.type, enabled=autocast):
        if isinstance(x, dict):
            x = {k: v[0] if isinstance(v, (tuple, list)) and len(v) == 1 else v for k, v in x.items()}
        else:
            if len(x) == 1:
                x = x[0]
        pred = model(x, **model_kwargs)

    pred = _to_list(pred)
    pred = _all_to(pred, torch.float32)

    return pred