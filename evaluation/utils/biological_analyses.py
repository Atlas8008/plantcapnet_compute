# This module requires a working parallel installation of R with the vegan package, in conjunction with the rpy2 package for python

from distutils.util import strtobool
import sys; sys.path.append("../../.."); sys.path.append("../../../..")

import os
import traceback
import numpy as np
import pandas as pd

from collections import defaultdict

from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


base = importr("base")

base.options(warn=1)

stats = importr("stats")
graphics = importr("graphics")
devices = importr("grDevices")
vegan = importr("vegan")


pandas2ri.activate()

SCALE_VALUES = {
    "IA_mean": {
        "Ach_mil": 3.41782407407407,
        "Cen_jac": 13.3773148148148,
        "Grasses": 10.3148148148148,
        "Lot_cor": 6.40740740740741,
        "Med_lup": 11.8325526932084,
        "Pla_lan": 11.5429234338747,
        "Sco_aut": 3.82242990654206,
        "Tri_pra": 33.3275462962963,
        "Dead_Litter": 14.1018518518519,
    },
    "IA_std": {
        "Ach_mil": 4.32344623717354,
        "Cen_jac": 10.2291038412395,
        "Grasses": 15.82698547125,
        "Lot_cor": 5.49315040605281,
        "Med_lup": 10.2218096848708,
        "Pla_lan": 10.1628548356455,
        "Sco_aut": 4.593933652136,
        "Tri_pra": 28.8047431885838,
        "Dead_Litter": 18.9018111629596,
    }
}

def dict_to_float(d):
    for k, v in d.items():
        if isinstance(v, dict):
            dict_to_float(v)
        else:
            d[k] = float(v)

def spearman_correlation(target, pred):
    val = stats.cor_test(target, pred, method="spearman")

    return val.rx2["estimate"]


def msae(target, pred, scale_kind, class_names, scale_values=None):
    if scale_kind not in ("none", "None", None):
        if scale_values is None:
            scale_values = [SCALE_VALUES[scale_kind][class_name] for class_name in class_names]
        else:
            if isinstance(scale_values, (int, float)):
                scale_values = [scale_values for class_name in class_names]
            else:
                scale_values = [scale_values[class_name] for class_name in class_names]

        scale_values = np.array(scale_values)[None]

        err = np.abs(target - pred) / scale_values
    else:
        err = np.abs(target - pred)

    return np.mean(err)


def dca(target, pred):
    """
    Performs a DCA on the target and predicted values.

    :param target: A matrix containing the target values.
    :param pred: A matrix containing the predicted values.
    :return: The DCA values for target and prediction in the format (target_dca1, target_dca2), (pred_dca1, pred_dca2).
    """
    dec_target = vegan.decorana(target)
    dec_pred = vegan.decorana(pred)

    target_dca1 = vegan.scores(dec_target)[..., 0]
    target_dca2 = vegan.scores(dec_target)[..., 1]

    pred_dca1 = vegan.scores(dec_pred)[..., 0]
    pred_dca2 = vegan.scores(dec_pred)[..., 1]

    return (target_dca1, target_dca2), (pred_dca1, pred_dca2)


def procrustes(target, pred):
    """
    Performs a procrustes analysis on the DCA values of the target and predicted values.

    :param target: A matrix containing the target values.
    :param pred: A matrix containing the predicted values.
    :return: The procrustes values.
    """
    dec_target = vegan.decorana(target)
    dec_pred = vegan.decorana(pred)

    proc = vegan.procrustes(dec_target, dec_pred)

    return proc




def procrustes_test(target, pred, dca=True, remove_empty_sites=False, scores="sites"):
    """
    Performs a procrustes test on the DCA values of the target and predicted values.

    :param target: A matrix containing the target values.
    :param pred: A matrix containing the predicted values.
    :return: The significance between the DCA transformed target and prediction values.
    """
    if remove_empty_sites:
        empty_sites = target.sum(axis=1) == 0

        pred = pred[~empty_sites]
        target = target[~empty_sites]

        print(f"Removed {empty_sites.sum()} empty sites, resulting shapes: {pred.shape}, {target.shape}")

    if dca:
        dec_target = vegan.decorana(target)
        dec_pred = vegan.decorana(pred)
    else:
        dec_target = target
        dec_pred = pred

    proc = vegan.protest(dec_target, dec_pred, scores=scores)

    return proc


def diversity(values):
    """Calculates the Shannon diversity for a matrix of populations using the vegan R package.

    Args:
        values (numpy.ndarray): A matrix containing the values to calculate the diversity for.
        The rows represent different populations and the columns represent different species. The values in the matrix should be non-negative integers representing the abundance of each species in each population. The function will return the diversity for each row (population) in the matrix. The diversity is calculated using the Shannon index, which takes into account both the richness and evenness of the species in each population.

    Returns:
        numpy.ndarray: A 1D array containing the diversity scores for each row (population) in the input matrix.
    """
    return vegan.diversity(values)


def get_data_as_array(fname, columns=None, plots=None, indices_fname=None):
    df = pd.read_csv(fname)

    if indices_fname:
        indices = pd.read_csv(indices_fname)
        df.index = pd.MultiIndex.from_frame(indices)

    # Reorder
    if columns is not None:
        df = df[columns]

    # Select specific plots
    if plots is not None:
        df = df.loc[plots]

    return df.to_numpy()


def get_pred_and_true_for_experiment(folder_name, experiment_name, columns, scale_predictions=True, scale_targets=True, plots=None):
    """Get the predicted and true values for a given experiment.

    Args:
        folder_name (str): The folder name where the experiment files are located.
        experiment_name (str): The name of the experiment.
        columns (list): The columns to select from the data files.
        scale_predictions (bool): Whether to scale the predictions.
        scale_targets (bool): Whether to scale the targets.
        plots (list): The plots to select from the data files if only a subset should be selected.
    Returns:
        tuple: The predicted and true values as numpy arrays.
    """
    indices_file = experiment_name + "_indices.csv"
    indices_file = os.path.join(folder_name, indices_file)

    print("Loading file for species set", columns)

    predictions_file = experiment_name + "_predictions.csv"
    predictions_file = os.path.join(folder_name, predictions_file)
    prediction_data = get_data_as_array(predictions_file, columns, plots=plots, indices_fname=indices_file)
    if scale_predictions:
        prediction_data *= 100

    targets_file = experiment_name + "_targets.csv"
    targets_file = os.path.join(folder_name, targets_file)
    target_data = get_data_as_array(targets_file, columns, plots=plots, indices_fname=indices_file)
    if scale_targets:
        target_data *= 100

    print("Read file", predictions_file, prediction_data.shape)
    print("Read file", targets_file, target_data.shape)

    return prediction_data, target_data

def log_and_print(key, val, tags, log_data):
    print(f"{key}: {val} ({tags})")
    log_data.append(f"{key}: {val} ({tags})")

def run_bio_analysis2(
        experiment_name,
        data_folder,
        evaluation_mode,
        split_names,
        include_dead_litter,
        run_dca,
        mode,
        scale_kind,
        log_file,
        logger_tags,
        class_names,
        plots=None,
        scale_values=None,
        file_prefix="",
        remove_empty_sites=True,
    ):
    """Run the biological evaluation analysis.

    Args:
        experiment_name (str): The name of the experiment.
        data_folder (str): The folder where the data files are located.
        evaluation_mode (str): The evaluation mode to use. Deprecated.
        split_names (list): The names of the splits to evaluate.
        include_dead_litter (bool): Whether to include dead litter in the analysis.
        run_dca (bool): Whether to run DCA on the data before performing Procrustes analysis.
        mode (str): The mode of evaluation to use.
        scale_kind (str): The kind of scaling to apply to the data.
        log_file (str): The file to log the results to. Deprecated.
        logger_tags (list): Tags for logging.
        class_names (list): The names of the classes to evaluate.
        plots (list): The plots to evaluate if only a subset should be evaluated.
        scale_values (dict): Scaling values for the classes.
        file_prefix (str): Prefix for the data files.
        remove_empty_sites (bool): Whether to remove empty sites from the data.
        """

    root_experiment_name = experiment_name

    folder_name = data_folder

    plant_names_target_order = class_names

    print()

    logger_tags.append(mode)

    metric_vals = defaultdict(list)
    metrics = {}

    metric_fns = {
        "spearman correlation": spearman_correlation,
    }

    log_data = [
        "## " + root_experiment_name + " ##",
        "Eval method: " + mode,
    ]

    if include_dead_litter:
        plant_names_target_order = plant_names_target_order + ["Dead_Litter"]
        log_data.append("Dead litter included")
    else:
        log_data.append("Dead litter *not* included")

    # Mean over all splits
    if mode == "mean":
        for split in split_names:
            metrics_split = {}
            metrics[f"_{split}"] = metrics_split

            df_mat, target_df_mat = get_pred_and_true_for_experiment(folder_name, split + file_prefix, plant_names_target_order, plots=plots)

            try:
                ptest = procrustes_test(
                    target_df_mat,
                    df_mat,
                    dca=run_dca,
                    remove_empty_sites=remove_empty_sites
                )

                metrics_split["dpc"] = ptest.rx2["t0"]
                metrics_split["significance"] = ptest.rx2["signif"]
            except RRuntimeError:
                traceback.print_exc()

                metrics_split["dpc"] = float("nan")
                metrics_split["significance"] = float("nan")

            metric_vals["dpc"].append(metrics_split["dpc"])

            div_pred = diversity(df_mat)
            div_target = diversity(target_df_mat)

            metrics_split["diversity predicted"] = np.mean(div_pred)
            metrics_split["diversity target"] = np.mean(div_target)
            metrics_split["diversity mae"] = np.mean(np.abs(div_pred - div_target))

            metric_vals["diversity predicted"].append(np.mean(metrics_split["diversity predicted"]))
            metric_vals["diversity target"].append(np.mean(metrics_split["diversity target"]))
            metric_vals["diversity mae"].append(metrics_split["diversity mae"])

            metrics_split[f"msae ({scale_kind})"] = msae(
                target_df_mat,
                df_mat,
                scale_kind,
                plant_names_target_order,
                scale_values=scale_values
            )

            metric_vals[f"msae ({scale_kind})"].append(metrics_split[f"msae ({scale_kind})"] )

            for metric_name, metric_fun in metric_fns.items():
                metrics_split[metric_name] = metric_fun(target_df_mat, df_mat)
                metric_vals[metric_name].append(metrics_split[metric_name])

            for metric_name, val in metrics_split.items():
                log_and_print(
                    metric_name,
                    float(val),
                    logger_tags + [split],
                    log_data,
                )

        for metric_name, vals in metric_vals.items():
            k = "mean " + metric_name
            v = float(np.mean(vals))

            metrics[k] = v
            log_and_print(
                k,
                v,
                logger_tags,
                log_data,
            )
    elif mode == "concat": # Concatenate results over all splits
        total_df_mat, total_target_df_mat = None, None

        for split in split_names:
            df_mat, target_df_mat = get_pred_and_true_for_experiment(folder_name, split + file_prefix, plant_names_target_order, plots=plots)

            if total_df_mat is None:
                total_df_mat = df_mat
                total_target_df_mat = target_df_mat
            else:
                total_df_mat = np.concatenate([total_df_mat, df_mat], axis=0)
                total_target_df_mat = np.concatenate([total_target_df_mat, target_df_mat], axis=0)

        try:
            ptest = procrustes_test(
                total_target_df_mat,
                total_df_mat,
                dca=run_dca,
                remove_empty_sites=remove_empty_sites,
            )

            metrics["dpc"] = ptest.rx2["t0"]
            metrics["significance"] = ptest.rx2["signif"]
        except RRuntimeError:
            traceback.print_exc()

            metrics["dpc"] = float("nan")
            metrics["significance"] = float("nan")

        div_pred = diversity(total_df_mat)
        div_target = diversity(total_target_df_mat)

        metrics["diversity predicted"] = np.mean(div_pred)
        metrics["diversity target"] = np.mean(div_target)
        metrics["diversity mae"] = np.mean(np.abs(div_pred - div_target))

        metrics[f"msae ({scale_kind})"] = msae(
            total_target_df_mat,
            total_df_mat,
            scale_kind,
            plant_names_target_order,
            scale_values=scale_values
        )
        metrics["#items"] = total_target_df_mat.shape[0]

        for metric_name, metric_fun in metric_fns.items():
            metrics[metric_name] = metric_fun(total_target_df_mat, total_df_mat)

        for metric_name, val in metrics.items():
            log_and_print(
                metric_name,
                float(val),
                logger_tags,
                log_data,
            )

    base.print(base.warnings())

    dict_to_float(metrics)

    return metrics


def bio_eval_full(total_df_mat, total_target_df_mat, plant_names_target_order, run_dca=True, scale_kind="IA_std", scale_values=None, remove_empty_sites=False):
    """Run the biological evaluation analysis.

    Args:
        total_df_mat (numpy.ndarray): The predicted values as a numpy array.
        total_target_df_mat (numpy.ndarray): The target values as a numpy array.
        plant_names_target_order (list): The names of the classes to evaluate.
        run_dca (bool): Whether to run DCA on the data before performing Procrustes analysis.
        scale_kind (str): The kind of scaling to apply to the data.
        scale_values (dict): Scaling values for the classes.
        remove_empty_sites (bool): Whether to remove empty sites from the data.
    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    metrics = {}

    remove_empty_sites = remove_empty_sites if isinstance(remove_empty_sites, bool) else strtobool(remove_empty_sites)
    run_dca = run_dca if isinstance(run_dca, bool) else strtobool(run_dca)

    if remove_empty_sites:
        empty_sites = total_target_df_mat.sum(axis=1) == 0

        total_df_mat = total_df_mat[~empty_sites]
        total_target_df_mat = total_target_df_mat[~empty_sites]

        print(f"Removed {len(empty_sites)} empty sites, resulting shapes: {total_df_mat.shape}, {total_target_df_mat.shape}")

    try:
        ptest = procrustes_test(
            total_target_df_mat,
            total_df_mat,
            dca=run_dca
        )

        metrics["dpc"] = float(ptest.rx2["t0"])
        metrics["significance"] = float(ptest.rx2["signif"])
    except RRuntimeError:
        traceback.print_exc()

        metrics["dpc"] = float("nan")
        metrics["significance"] = float("nan")

    div_pred = diversity(total_df_mat)
    div_target = diversity(total_target_df_mat)

    metrics["diversity predicted"] = np.mean(div_pred)
    metrics["diversity target"] = np.mean(div_target)
    metrics["diversity mae"] = np.mean(np.abs(div_pred - div_target))

    metrics[f"msae ({scale_kind})"] = msae(
        total_target_df_mat,
        total_df_mat,
        scale_kind,
        plant_names_target_order,
        scale_values=scale_values
    )
    metrics["#items"] = total_target_df_mat.shape[0]

    print(metrics)
    return metrics


def get_data_files_as_array(fnames, columns=None):
    if not isinstance(fnames, (tuple, list)):
        fnames = [fnames]

    arrays = [get_data_as_array(fname, columns) for fname in fnames]

    if len(arrays) == 1:
        return arrays[0]

    return np.concatenate(arrays, axis=0)


def bio_eval_from_file(pred_file, target_file, scale_predictions=True, scale_targets=True, class_names=None, **kwargs):
    if class_names is None:
        class_names = pd.read_csv(pred_file).columns

    prediction_data = get_data_files_as_array(pred_file, class_names)
    if scale_predictions:
        prediction_data *= 100

    target_data = get_data_files_as_array(target_file, class_names)
    if scale_targets:
        target_data *= 100

    metrics = bio_eval_full(
        prediction_data,
        target_data,
        plant_names_target_order=class_names,
        **kwargs,
    )

    return metrics

def _maybe_set_value(args, k, v):
    if k is not None:
        if not v:
            raise ValueError("No value provided for key " + str(k))
        if len(v) == 1:
            v = v[0]
        args[k] = v
        k = None
    elif v:
        raise ValueError("No key provided for value " + str(v))

def _get_args():
    import sys

    arg_list = sys.argv[1:]

    args = {}

    k = None
    v = []

    for item in arg_list:
        if item.startswith("--"):
            _maybe_set_value(args, k, v)

            v = []

        if item.startswith("--"):
            k = item[2:]
        else:
            v.append(item)

    _maybe_set_value(args, k, v)

    return args


if __name__ == "__main__":
    import json

    args = _get_args()

    print("Args:")
    print(args)

    if "output_file" in args:
        output_file = args["output_file"]
        del args["output_file"]
    else:
        output_file = "results.json"

    metrics = bio_eval_from_file(
        **args
    )

    with open(output_file, "w") as f:
        f.write(json.dumps(metrics, indent=True))