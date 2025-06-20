import pandas as pd


def correlation_metrics_from_files(predictions_file, targets_file, indices_file, suffix=None, names=None, **kwargs):

    print(predictions_file)

    aggregations = {
        "": "",
        ("site", "year", "month", "day"): "mean:day",
    }

    if not isinstance(predictions_file, list):
        predictions_file = [predictions_file]
    if not isinstance(targets_file, list):
        targets_file = [targets_file]
    if not isinstance(indices_file, list):
        indices_file = [indices_file]

    if names is None:
        names = ["set"] * len(predictions_file)
    else:
        assert len(names) == len(predictions_file)

    predictions_full = None
    targets_full = None
    indices_full = None

    metrics_final = {}

    for name, p_file, t_file, i_file in zip(names, predictions_file, targets_file, indices_file):
        print(f"{name}:")

        predictions = pd.read_csv(p_file)
        targets = pd.read_csv(t_file)
        indices = pd.read_csv(i_file)

        if predictions_full is None:
            predictions_full = predictions
            targets_full = targets
            indices_full = indices
        else:
            predictions_full = pd.concat(
                [predictions_full, predictions], axis="rows")
            targets_full = pd.concat(
                [targets_full, targets], axis="rows")
            indices_full = pd.concat(
                [indices_full, indices], axis="rows")

        metrics = get_dataset_metrics(
            predictions=predictions,
            targets=targets,
            indices=indices,
            suffix=suffix,
            aggregations=aggregations,
            **kwargs,
        )

        metrics_final[name] = metrics

    mean_df = pd.concat([pd.DataFrame(d, index=[n]) for n, d in metrics_final.items()])
    metrics_mean = mean_df.mean(axis="rows").to_dict()

    metrics_concat = get_dataset_metrics(
        predictions=predictions_full,
        targets=targets_full,
        indices=indices_full,
        suffix=suffix,
        aggregations=aggregations,
        **kwargs,
    )

    metrics_final["concat"] = metrics_concat
    metrics_final["mean"] = metrics_mean

    print(f"Concat:")
    print(metrics_final["concat"])

    print("Mean:")
    print(metrics_final["mean"])

    return metrics_final


def get_dataset_metrics(predictions, targets, indices, suffix, aggregations, class_names=None, plots=None):

    predictions_w_idx = predictions.copy()
    targets_w_idx = targets.copy()

    predictions_w_idx.index = pd.MultiIndex.from_frame(indices)
    targets_w_idx.index = pd.MultiIndex.from_frame(indices)

    if class_names:
        predictions_w_idx = predictions_w_idx[class_names]
        targets_w_idx = targets_w_idx[class_names]

    if plots is not None:
        predictions_w_idx = predictions_w_idx.loc[plots]
        targets_w_idx = targets_w_idx.loc[plots]

    if suffix is not None:
        suffix = f" ({suffix})"

    metrics = {}

    for agg, agg_name in aggregations.items():
        if agg:
            agg_name = "_" + agg_name

            if isinstance(agg, tuple):
                agg = list(agg)

            pred = predictions_w_idx.groupby(agg).mean()
            targ = targets_w_idx.groupby(agg).mean()
        else:
            pred = predictions_w_idx
            targ = targets_w_idx

        metrics[f"Correlation{suffix}" + agg_name] = targ.corrwith(pred).mean()
        metrics[f"CorrelationC{suffix}" + agg_name] = targ.corrwith(pred, axis="columns").mean()
        metrics[f"CorrelationZ{suffix}" + agg_name] = targ.corrwith(pred).fillna(0).mean()
        metrics[f"CorrelationZC{suffix}" + agg_name] = targ.corrwith(pred, axis="columns").fillna(0).mean()
        metrics[f"MSE{suffix}" + agg_name] = ((targ - pred) ** 2).mean().mean()
        metrics[f"MAE{suffix}" + agg_name] = (targ - pred).abs().mean().mean()

    return metrics
