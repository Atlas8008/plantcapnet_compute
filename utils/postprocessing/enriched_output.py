import math
import torch
from torch import nn


__all__ = ["interlaced_tile_prediction", "predict_enriched_output_by_augmentation"]

def _ensure_list(val):
    if not isinstance(val, (tuple, list)):
        return [val]

    return val

def _maybe_drop_list(val):
    if len(val) == 1:
        return val[0]

    return val

def _maybe_resize_all(val, resize):
    if resize is not None:
        val = _ensure_list(val)

        val = [
            nn.functional.interpolate(
                v,
                size=resize,
                mode="bilinear")
            for v in val
        ]
        val = _maybe_drop_list(val)

    return val

def _zip_map(l, fun):
    out_l = []

    for sublist in zip(*l):
        out_l.append(fun(sublist))

    return out_l

# Device wrap
def dw(device):
    if device != -1:
        return device
    return "cpu"


def interlaced_tile_prediction(x: torch.Tensor, prediction_fn, tile_size=256, interlace_type="simple", autoresize=False, **kwargs):
    b, c, h, w = x.shape

    # Half Tile Size
    hts = tile_size // 2
    output_tile_size = None
    ohts = None
    output_step_size = None
    oh = None
    ow = None

    output = None
    counts = None

    def lazy_gen_output_and_counts(c, example_output):
        example_output = example_output[0]

        output_tile_size = example_output.shape[2]
        ohts = output_tile_size // 2
        factor = output_tile_size / tile_size

        assert example_output.shape[2] == example_output.shape[3], "Output dimensions should have the same size"

        if interlace_type == "simple":
            output_step_size = output_tile_size
        elif interlace_type == "double":
            output_step_size = ohts
        else:
            raise ValueError()

        output = [torch.zeros((
            b,
            channels,
            int(factor * h) + ohts + output_tile_size,
            int(factor * w) + ohts + output_tile_size,
        )).to(dw(x.device)) for channels in c]

        counts = torch.zeros_like(output[0]).to(dw(x.device))

        return output, counts, output_tile_size, ohts, output_step_size

    x = nn.functional.pad(x, (0, tile_size, 0, tile_size))

    if interlace_type == "simple":
        step_size = tile_size
    elif interlace_type == "double":
        step_size = hts
    else:
        raise ValueError("Invalid interlace type: " + interlace_type)

    # Standard tiling with optional step_size interlacing
    for row in range(math.ceil(h / step_size)):
        for col in range(math.ceil(w / step_size)):
            out = prediction_fn(x[:, :,
                    row * step_size:(row * step_size) + tile_size,
                    col * step_size:(col * step_size) + tile_size,
                ],
                **kwargs
            )

            # Handle everything as list to be able to handle multiple outputs
            out = _ensure_list(out)

            # Generate output with correct shape lazily by checking the size of the first output
            if output is None:
                output, counts, output_tile_size, ohts, output_step_size = \
                    lazy_gen_output_and_counts([o.shape[1] for o in out], out)

            #print("out shape:", out.shape)
            for o, o_pred in zip(output, out):
                o[:, :,
                    row * output_step_size + ohts:
                        (row * output_step_size) + output_tile_size + ohts,
                    col * output_step_size + ohts:
                        (col * output_step_size) + output_tile_size + ohts,
                ] = o_pred

            counts[:, :,
                row * output_step_size + ohts:
                    (row * output_step_size) + output_tile_size + ohts,
                col * output_step_size + ohts:
                    (col * output_step_size) + output_tile_size + ohts,
            ] += 1

    x = nn.functional.pad(x, (hts, 0, hts, 0))

    # 1st degree interlaced tiling (tiles with previous corners in the center)
    for row in range(math.ceil(h / step_size) + 1):
        for col in range(math.ceil(w / step_size) + 1):
            out = prediction_fn(x[:, :,
                    row * step_size:(row * step_size) + tile_size,
                    col * step_size:(col * step_size) + tile_size,
                ],
                **kwargs
            )

            # Handle everything as list to be able to handle multiple outputs
            out = _ensure_list(out)

            if autoresize:
                out = [
                    nn.functional.interpolate(
                        out_item,
                        size=(tile_size, tile_size),
                        mode="bilinear"
                    )
                    for out_item in out
                ]

            for o, o_pred in zip(output, out):
                o[:, :,
                    row * output_step_size:
                        (row * output_step_size) + output_tile_size,
                    col * output_step_size:
                        (col * output_step_size) + output_tile_size,
                ] += o_pred

            counts[:, :,
                row * output_step_size:
                    (row * output_step_size) + output_tile_size,
                col * output_step_size:
                    (col * output_step_size) + output_tile_size,
            ] += 1

    final_outputs = []
    factor = output_tile_size / tile_size

    for out in output:
        o = out / counts

        # Crop output to original size
        o = o[:, :, ohts:ohts + int(factor * h), ohts:ohts + int(factor * w)]

        final_outputs.append(o)

    return _maybe_drop_list(final_outputs)

@torch.no_grad()
def predict_enriched_output_by_augmentation(x, model, device, transforms=None, scales=(1.0, ), vertical_flips=False, horizontal_flips=False, interlaced=False, interlace_tile_size=256, interlace_type="simple", interlace_autoresize=False, model_mode=None, resize_inputs_to_multiples_of=None, aggregation="mean", model_kwargs=None):
    assert aggregation in ("softmax", "mean", "max"), "Invalid aggregation type: " + aggregation
    model_kwargs = model_kwargs or {}

    if transforms is not None:
        # In case a raw image is input, we first need to transform it and generate a batch with batch size 1
        x = transforms(x)
        x_orig = x[None].to(dw(device))
        is_batch = False
    else:
        x_orig = x.to(dw(device))
        is_batch = len(x_orig.shape) >= 4

    if not isinstance(scales, (tuple, list)):
        scales = (scales, )

    x_vals = []
    preds = []

    for scale in scales:
        x_val = nn.functional.interpolate(
            x_orig, scale_factor=scale, mode="bilinear")

        if resize_inputs_to_multiples_of is not None:
            orig_size = x_val.shape[-2:]
            r_val = resize_inputs_to_multiples_of

            x_val = nn.functional.interpolate(
                x_val,
                size=(
                    r_val * math.ceil(orig_size[0] / r_val),
                    r_val * math.ceil(orig_size[1] / r_val),
                ),
                mode="bilinear"
            )

        x_vals.append(x_val)

    if vertical_flips:
        v_flipped = [torch.flip(x, dims=[-2]) for x in x_vals]
        v_flips = [False] * len(x_vals) + [True] * len(x_vals)
        x_vals += v_flipped
    else:
        v_flips = [False] * len(x_vals)

    if horizontal_flips:
        h_flipped = [torch.flip(x, dims=[-1]) for x in x_vals]
        v_flips += v_flips
        h_flips = [False] * len(x_vals) + [True] * len(x_vals)
        x_vals += h_flipped
    else:
        h_flips = [False] * len(x_vals)

    def pred_fn(x, resize=None):
        if model_mode is not None:
            pred = model(x, mode=model_mode, **model_kwargs)
        else:
            pred = model(x, **model_kwargs)

        return _maybe_resize_all(pred, resize=resize)

    # For memory optimization
    final_pred = None
    pred_count = 0
    resize = None

    for x, vflipped, hflipped in zip(x_vals, v_flips, h_flips):
        with torch.no_grad(), torch.amp.autocast(device.type):
            # Optionally apply interlacing
            if not interlaced:
                pred = pred_fn(x, resize=resize)
            else:
                if not isinstance(interlace_tile_size, (tuple, list)):
                    interlace_tile_size = [interlace_tile_size]

                # For memory optimization
                preds_agg = None
                sub_pred_count = 0

                # List for unoptimized version
                #sub_preds = []

                for its in interlace_tile_size:
                    pred = interlaced_tile_prediction(
                        x,
                        pred_fn,
                        tile_size=its,
                        interlace_type=interlace_type,
                        autoresize=interlace_autoresize,
                    )

                    pred = _maybe_resize_all(pred, resize=resize)
                    pred = _ensure_list(pred)

                    #sub_preds.append(pred)
                    if preds_agg is None:
                        preds_agg = pred
                    else:
                        preds_agg = [a + b for a, b in zip(preds_agg, pred)]

                    sub_pred_count += 1

                pred = [v / sub_pred_count for v in preds_agg]

                if resize is None:
                    resize =  pred[0].shape[-2:]

                # Average over multiple interlace outputs
                # pred = _zip_map(
                #     sub_preds,
                #     lambda l: torch.mean(torch.stack(l), dim=0))

            pred = _ensure_list(pred)

            # Flip everything back
            if vflipped:
                pred = map(lambda v: torch.flip(v, dims=[-2]), pred)
            if hflipped:
                pred = map(lambda v: torch.flip(v, dims=[-1]), pred)

            pred = list(pred)

            pred_count += 1

            if final_pred is None:
                final_pred = pred
            else:
                if aggregation == "mean":
                    final_pred = [a + b for a, b in zip(final_pred, pred)]
                elif aggregation == "max":
                    final_pred = [torch.maximum(a, b) for a, b in zip(final_pred, pred)]
                else:
                    preds.append(pred)

    if pred_count > 1:
        if aggregation == "mean":
            pred = [v / pred_count for v in final_pred]
        elif aggregation == "softmax":
            pred = _zip_map(final_pred, lambda x: torch.sum(torch.softmax(x, dim=0) * x, dim=0))
        elif aggregation == "max":
            pred = final_pred
        else:
            raise ValueError("Invalid aggregation method: " + aggregation)
        # def combine_fun(preds):
        #     preds = torch.stack(preds)
        #     if aggregation == "mean":
        #         pred = torch.mean(preds, dim=0)
        #     elif aggregation == "softmax":
        #         pred = torch.sum(torch.softmax(preds, dim=0) * preds, dim=0)
        #     elif aggregation == "max":
        #         pred = torch.amax(preds, dim=0)

        #     return pred

        # pred = _zip_map(preds, combine_fun)
    else:
        # Simply remove list
        pred = preds[0]

    # Remove batch dimension if input was only a single image without batch dimension from the beginning
    if not is_batch:
        pred = list(map(lambda p: p[0], pred))

    return _maybe_drop_list(pred)


