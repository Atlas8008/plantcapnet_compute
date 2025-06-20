import os
import json
import torch
import numpy as np
import pandas as pd
import torchmetrics

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict

from utils.augmentations import Preprocess, DePreprocess
from utils.torch_utils.multiio import *
from utils.display_utils import display_segmentation

import models
import cap_utils as lu

from .evaluation import EvaluationMethod, PostEvaluationMethod


def merge_dead_litter_into_background(tensor, class_names):
    dl_idx = class_names.index("Dead Litter")
    bg_idx = class_names.index("Background")

    tensor[tensor == dl_idx] = bg_idx

    # Reduce all indices after dead litter by 1
    tensor = torch.where(
        tensor >= dl_idx,
        tensor - 1,
        tensor
    )

    return tensor


class CoverPredictionEvaluation(EvaluationMethod):
    def __init__(self, name, data_loader, metrics, split_name, verbose=False, logger_tags=None, predictions_folder=None, output_index=0, multivalue_names=None, avg_modes=None, model_kwargs=None):
        """
        Args:
            name (str): Name/title of this evaluation process for later differentiation.
            data_loader (DataLoader): DataLoader for the dataset to evaluate.
            metrics (list): List of metrics to evaluate.
            split_name (str): Name of the dataset split (e.g., "train", "val", "test").
            verbose (bool): Whether to print detailed results.
            logger_tags (list): List of tags for logging.
            predictions_folder (str): Folder to save predictions.
            output_index (int): Index of the output in the model's output tuple.
            multivalue_names (dict): Dictionary mapping multivalue indices to names.
            avg_modes (dict): Dictionary mapping average modes to their keys.
        """
        super().__init__(name)
        self.data_loader = data_loader
        self.metrics = metrics
        self.split_name = split_name
        self.verbose = verbose

        self.multivalue_names = multivalue_names
        self.output_index = output_index
        self.predictions_folder = predictions_folder
        self.logger_tags = logger_tags
        self.avg_modes = avg_modes
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def __call__(self, model, device):
        model = model.to(device)

        predictions = []

        for metric in self.metrics:
            metric.reset()
            metric.to(device)

        model.eval()

        print(f"Evaluating {self.name} model:")
        print(json.dumps(self.model_kwargs, indent=True))

        for i, t in enumerate(tqdm(self.data_loader)):
            x, y, _ = _unpack_tuple(t, device)
            y = y[self.output_index]

            with torch.amp.autocast(device.type), torch.no_grad():
                pred = _predict(
                    x,
                    model,
                    **self.model_kwargs,
                )
                x = None
                torch.cuda.empty_cache()

                if isinstance(pred, (tuple, list)):
                    pred = pred[self.output_index]

            for metric in self.metrics:
                metric.update(pred, y)

            for prediction in pred:
                predictions.append(prediction.cpu().numpy())

        test_results = {
            self.name + "/" + (metric.name if hasattr(metric, "name") and metric.name else metric.__class__.__name__):
                metric.compute().item() for metric in self.metrics
        }

        self.maybe_print_results(test_results)
        self.maybe_log_predictions(predictions)

        return test_results

    def maybe_log_predictions(self, predictions):
        if self.predictions_folder:
            if self.multivalue_names is not None:
                multivalue_names = self.multivalue_names
            else:
                multivalue_names = {None: ""}

            for mv_idx, mv_name in multivalue_names.items():
                if mv_name:
                    mv_name = "_" + mv_name

                lu.results_to_file(
                    data=predictions,
                    folder=self.predictions_folder,
                    split=self.split_name,
                    dataset=self.data_loader.dataset,
                    prefix=("_" + self.name if self.name else "") + mv_name,
                    output_index=self.output_index,
                    multivalue_index=mv_idx,
                    avg_modes=self.avg_modes,
                )

    def maybe_print_results(self, test_results):
        if self.verbose:
            print("Test results:")
            print(test_results)


class TimeSeriesCoverPredictionEvaluation(CoverPredictionEvaluation):
    def __init__(self, *args, ts_prediction_model, training_data_loader=None, train_ts_model="auto", intermediate_avg_key=None, n_supplementary_time_steps=0, **kwargs):
        """
        Time series evaluation method for cover prediction.

        Args:
            ts_prediction_model (TrainableTimeSeriesModel): Time series prediction model.
            training_data_loader (DataLoader): DataLoader for the training dataset.
            train_ts_model (bool): Whether to train/fit the time series model before evaluation using the training_data_loader.
            intermediate_avg_key (str): Key for averaging intermediate results. E.g., "day", if the data is hourly and the time series model is trained on daily averages.
            n_supplementary_time_steps (int): Number of supplementary time steps to consider. For time-step cover prediction.
            kwargs: Additional arguments for the model.
        """
        super().__init__(*args, **kwargs)

        self.ts_prediction_model = ts_prediction_model
        self.training_data_loader = training_data_loader
        self.train_ts_model = train_ts_model
        self.intermediate_avg_key = intermediate_avg_key
        self.n_supplementary_time_steps = n_supplementary_time_steps

    def __call__(self, model, device):
        if self.train_ts_model == "auto":
            train_ts_model = isinstance(self.ts_prediction_model, models.time_series.TrainableTimeSeriesModel)
        else:
            train_ts_model = self.train_ts_model

        print("Applying time series model:")
        print(self.ts_prediction_model)

        if train_ts_model and self.ts_prediction_model.requires_training:
            # Extract training data
            print("Extracting time series training data")
            t_predictions, t_targets, t_indices = self._predict(
                model, device, self.training_data_loader, apply_ts_model=False, group_outputs=True, filter_gt=False, apply_metrics=False,
            )

            print("Train seq ids:", self.training_data_loader.dataset.indices_to_seq_ids(t_indices))
            self.ts_prediction_model.fit(
                t_predictions,
                t_targets,
                self.training_data_loader.dataset.indices_to_seq_ids(t_indices),
            )

        print(f"Evaluating {self.name} model:")
        print(json.dumps(self.model_kwargs, indent=True))

        predictions, targets, indices = self._predict(
            model, device, self.data_loader, apply_ts_model=True, filter_gt=True, apply_metrics=True,
        )

        test_results = {
            self.name + "/" + (metric.name if hasattr(metric, "name") and metric.name else metric.__class__.__name__):
                metric.compute().item() for metric in self.metrics
        }

        self.maybe_print_results(test_results)
        self.maybe_log_pti(predictions, targets, indices)

        return test_results

    def _predict(self, model, device, data_loader, apply_ts_model=False, group_outputs=False, filter_gt=False, apply_metrics=True, detach_predictions=True):
        model = model.to(device)

        predictions = []
        targets = []
        indices = []

        for metric in self.metrics:
            metric.reset()
            metric.to(device)

        model.eval()

        ts_ordered_indices = data_loader.dataset.source_indices
        ts_data_is_ground_truth_lists = data_loader.dataset.data_is_ground_truth

        assert len(ts_data_is_ground_truth_lists) == len(ts_ordered_indices)

        print("Lists:", len(ts_data_is_ground_truth_lists), len(ts_ordered_indices), len(data_loader))

        print("Dataset length done")

        for i in tqdm(range(len(data_loader.dataset))):
            ts_x, ts_y = data_loader.dataset[i]

            print(i)
            print(len(ts_ordered_indices))

            ts_indices = ts_ordered_indices[i]
            ts_data_is_ground_truth = ts_data_is_ground_truth_lists[i]
            ts_predictions = []
            ts_targets = []

            for ts_idx, t in enumerate(zip(ts_x, ts_y)):
                x, y, _ = _unpack_tuple(t, device)
                is_ground_truth = ts_data_is_ground_truth[ts_idx]

                if not isinstance(self.model_kwargs, (tuple, list)):
                    model_kwargs = [self.model_kwargs]
                else:
                    model_kwargs = self.model_kwargs

                with torch.amp.autocast(device.type), torch.no_grad():
                    if self.n_supplementary_time_steps > 0:
                        supplementary_ts = ts_predictions[-1:-1 - self.n_supplementary_time_steps:-1]

                        if len(supplementary_ts) > 0:
                            # If only a single step, input that step as tensor
                            if self.n_supplementary_time_steps == 1:
                                supplementary_ts = torch.tensor(supplementary_ts[0])
                            else: # Otherwise format the multiple steps as a time series, i.e., stack on a new dimension
                                supplementary_ts = torch.stack(
                                    [torch.tensor(v) for v in supplementary_ts],
                                    model.temporal_dim,
                                )

                            supplementary_ts = supplementary_ts.to(device)

                            x = {
                                "x": x,
                                "suppl_data": [supplementary_ts]
                            }

                            print("Basing prediction on ", supplementary_ts)


                    pred_components = []

                    for mk in model_kwargs:
                        pred_component = _predict(
                            x,
                            model,
                            **mk,
                        )

                        if isinstance(pred_component, (tuple, list)):
                            pred_component = pred_component[self.output_index]

                        if detach_predictions:
                            pred_component = pred_component.detach()

                        pred_components.append(pred_component)

                    print("Predicted", pred_components)

                    pred = pred_components

                    torch.cuda.empty_cache()

                    if isinstance(y, (tuple, list)):
                        targ = y[self.output_index]
                    else:
                        targ = y

                # Here, we have a list of prediction items:
                # [shape(b, kwarg_output1), shape(b, kwarg_output2), ...]

                # Interpret all batch items as time series elements and append to ts list
                for pt in zip(*pred):
                    pt = [pt_component.cpu().numpy() for pt_component in pt]
                    ts_predictions.append(pt)
                for target in targ:
                    ts_targets.append(target.cpu().numpy())

            if apply_ts_model and self.ts_prediction_model is not None:
                # Reformat ts_predictions to fit time series format
                ts_predictions = tuple(list(v) for v in zip(*ts_predictions))

                print("INTERMEDIATE AVG KEY:", self.intermediate_avg_key)

                if self.intermediate_avg_key:
                    ts_indices_orig = ts_indices
                    ts_predictions, ts_targets, ts_indices = lu.average_over_index_key(
                        self.intermediate_avg_key,
                        predictions=ts_predictions,
                        targets=ts_targets,
                        indices=ts_indices,
                        id_headers=data_loader.dataset.index_names,
                    )
                    # We need to "average" the gt_list too in order to be consistent with the transformed averaged lists
                    ts_data_is_ground_truth = lu.average_gt_list_over_index_key(
                        self.intermediate_avg_key,
                        indices=ts_indices_orig,
                        id_headers=data_loader.dataset.index_names,
                        gt_list=ts_data_is_ground_truth,
                    )
                    seq_ids = list(range(len(ts_predictions)))
                else:
                    seq_ids = data_loader.dataset.indices_to_seq_ids(ts_indices)

                print("Eval seq ids:", seq_ids)
                ts_predictions = self.ts_prediction_model(
                    ts_predictions,
                    seq_ids=seq_ids,
                    clip=(0.0, 1.0),
                )
            else:
                # Only the first prediction component is a valid prediction, if no time-series model is applied
                ts_predictions = [p[0] for p in ts_predictions]

            if filter_gt:
                print("Before filtering:", len(ts_predictions), len(ts_targets), len(ts_indices))

                # Filter according to gt
                ts_predictions = [p for p, is_gt in zip(ts_predictions, ts_data_is_ground_truth) if is_gt]
                ts_targets = [t for t, is_gt in zip(ts_targets, ts_data_is_ground_truth) if is_gt]
                ts_indices = [vidx for vidx, is_gt in zip(ts_indices, ts_data_is_ground_truth) if is_gt]

                print("After filtering:", len(ts_predictions), len(ts_targets), len(ts_indices))
            else:
                print("Data length:", len(ts_predictions), len(ts_targets), len(ts_indices))

            if apply_metrics:
                for p, t, i, is_gt in zip(ts_predictions, ts_targets, ts_indices, ts_data_is_ground_truth):
                    if is_gt:
                        for metric in self.metrics:
                            metric.update(
                                torch.tensor(p)[None].to(device),
                                torch.tensor(t)[None].to(device),
                            )

            if group_outputs:
                predictions.append(ts_predictions)
                targets.append(ts_targets)
                indices.append(ts_indices)
            else:
                for prediction in ts_predictions:
                    predictions.append(prediction)
                for target in ts_targets:
                    targets.append(target)
                for index in ts_indices:
                    indices.append(index)

        print(f"{len(predictions)} final predictions")

        return predictions, targets, indices

    def maybe_log_pti(self, predictions, targets, indices):
        dataset = self.data_loader.dataset

        if self.predictions_folder:
            if self.multivalue_names is not None:
                multivalue_names = self.multivalue_names
            else:
                multivalue_names = {None: ""}

            for mv_idx, mv_name in multivalue_names.items():
                if mv_name:
                    mv_name = "_" + mv_name

                headers = dataset.class_names
                id_headers = dataset.index_names

                prefix = ("_" + self.name if self.name else "") + mv_name

                lu.pti_to_file(
                    predictions=predictions,
                    targets=targets,
                    indices=indices,
                    headers=headers,
                    id_headers=id_headers,
                    folder=self.predictions_folder,
                    file_prefix=self.split_name + prefix,
                    output_index=self.output_index,
                    multivalue_index=mv_idx,
                )


                if self.avg_modes:
                    for k, v in self.avg_modes.items():
                        avg_prefix = k

                        lu.pti_to_file(
                            predictions=predictions,
                            targets=targets,
                            indices=indices,
                            headers=headers,
                            id_headers=id_headers,
                            folder=self.predictions_folder,
                            file_prefix=self.split_name + prefix + "_" + avg_prefix,
                            output_index=self.output_index,
                            multivalue_index=mv_idx,
                            avg_key=v,
                        )


class CoverSegmentationEvaluation(EvaluationMethod):
    def __init__(self, name, image_size_hw, include_dead_litter, normalization, verbose=True, logger_tags=None, input_index=0, output_index=0, segmentation_save_path=None, error_map_save_path=None, model_kwargs=None, ):
        """
        Evaluation method for cover segmentation.

        Args:
            name (str): Name/title of this evaluation process for later differentiation.
            image_size_hw (tuple): Image size for resizing.
            include_dead_litter (bool): Whether to include the dead litter class in the evaluation.
            normalization (str): Normalization to apply to images (tf, torch, caffe).
            verbose (bool): Whether to print detailed results.
            logger_tags (list): List of tags for logging.
            input_index (int): Index of the input in the model's input tuple. Only relevant for multi-input models.
            output_index (int): Index of the output in the model's output tuple.
            segmentation_save_path (str): Path to save segmentation maps.
            error_map_save_path (str): Path to save error maps.
            model_kwargs (dict): Additional arguments for the model.
        """
        super().__init__(name)

        self.image_size = image_size_hw
        self.normalization = normalization
        self.include_dead_litter = include_dead_litter
        self.segmentation_save_path = segmentation_save_path
        self.error_map_save_path = error_map_save_path

        self.verbose = verbose

        self.input_index = input_index
        self.output_index = output_index
        self.logger_tags = logger_tags
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def __call__(self, model, device):
        # Transforms
        transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            Preprocess(self.normalization),
        ])

        dataset = None # Enter custom dataset with images and segmentations here to evaluate
        dataset_loader = DataLoader(dataset, num_workers=8)

        model = model.to(device)

        model.eval()

        class_names_nodl = dataset.class_names
        class_names_dl = dataset.class_names_dl

        if not self.include_dead_litter:
            segmentation_class_names = class_names_nodl
        else:
            segmentation_class_names = class_names_dl

        print("Segmentation class names:", segmentation_class_names)

        # len(class_names) + 1 due to uncertainty

        with torch.no_grad():
            miou = torchmetrics.JaccardIndex(
                task="multiclass",
                num_classes=len(segmentation_class_names) + 1
            )

            for i, t in enumerate(dataset_loader):
                print("Image", i + 1)

                x, gt, _ = _unpack_tuple(t, device)

                pred = _predict(
                    x,
                    model,
                    **self.model_kwargs
                )

                if isinstance(pred, (tuple, list)):
                    pred = pred[self.output_index]
                    gt = gt[self.output_index]

                gt = gt[0].cpu()

                if not self.include_dead_litter:
                    gt = merge_dead_litter_into_background(
                        gt,
                        class_names=class_names_dl
                    )

                if pred.shape != gt.shape:
                    # Bilinear resizing to gt map
                    print("Resizing", pred.shape, "to", gt.shape)
                    pred = nn.functional.interpolate(pred, gt.shape[:2], mode="bilinear")

                x_resized = nn.functional.interpolate(x[self.input_index], gt.shape[:2], mode="bilinear")

                pred = pred[0].to(torch.float32)

                pred = torch.moveaxis(pred.cpu(), 0, -1)
                pred = torch.argmax(pred, dim=-1)

                # Increase class index of every class by 1 to be on par with the GT and avoid the uncertainty class
                pred += 1

                if self.segmentation_save_path is not None:
                    self.save_segmap(pred - 1, x_resized[0], segmentation_class_names, i)
                if self.error_map_save_path is not None:
                    self.save_error_map(pred, gt, x_resized[0], i)

                miou.update(pred.to(torch.int32), gt.to(torch.int32))

            cm = miou.confmat

            # Remove uncertainty class (black areas in image)
            cm = cm[1:, 1:]

            true_pos = torch.diag(cm)
            row_sums = torch.sum(cm, dim=0)
            col_sums = torch.sum(cm, dim=1)

            numerator = true_pos
            denominator = row_sums + col_sums - true_pos

            iou_vals = numerator / denominator

            iou_vals = [v.item() for v in iou_vals]

            segmentation_class_names.extend(["Total", "Plantsonly"])
            iou_vals.extend([np.mean(iou_vals), np.mean(iou_vals[:-2])])

            if self.verbose:
                print("IoU vals:")
                for cname, val in zip(segmentation_class_names, iou_vals):
                    print(cname, val)

        return dict(zip(segmentation_class_names, iou_vals))

    def save_segmap(self, pred, x, class_names, id):
        pred = pred.cpu().numpy()
        x = DePreprocess(self.normalization)(x.cpu())
        x = torch.moveaxis(x, 0, -1).cpu().numpy().astype("uint8")

        os.makedirs(self.segmentation_save_path, exist_ok=True)

        print(pred.shape, x.shape)

        display_segmentation(
            segmentation_map=pred,
            image=x,
            class_name_list=class_names,
            save_to=os.path.join(self.segmentation_save_path, f"{id}.jpg"),
            do_show=False,
            close_fig=True,
        )

    def save_error_map(self, pred, gt, x, id):
        pred = pred.cpu().numpy()
        x = DePreprocess(self.normalization)(x.cpu())
        x = torch.moveaxis(x, 0, -1).cpu().numpy().astype("uint8")

        print(pred.shape, gt.shape, x.shape)

        correct_map = np.equal(pred, gt)
        incorrect_map = np.not_equal(pred, gt)

        uncertainty_map = gt == 0
        correct_map[np.where(uncertainty_map)] = 0
        incorrect_map[np.where(uncertainty_map)] = 0

        print("Before stack:", uncertainty_map.shape, correct_map.shape, incorrect_map.shape)

        maps = np.stack([uncertainty_map, correct_map, incorrect_map], axis=-1)

        print(np.sum(maps, axis=(0, 1)))

        os.makedirs(self.error_map_save_path, exist_ok=True)

        display_segmentation(
            segmentation_map=np.argmax(maps, axis=-1),
            image=x,
            class_name_list=["Uncertainty", "Correct", "Incorrect"],
            colormap=[(0.5, 0.5, 0.5), (0, 1.0, 0), (1.0, 0, 0)],
            save_to=os.path.join(self.error_map_save_path, f"{id}.jpg"),
            do_show=False,
            close_fig=True,
        )



class MetricAggregation(PostEvaluationMethod):
    def __init__(self, name, metric_names, folder, subfolder_names) -> None:
        super().__init__(name)

        if not isinstance(metric_names, list):
            metric_names = [metric_names]

        self.metric_names = metric_names
        self.folder = folder
        self.subfolders = subfolder_names

    def __call__(self):
        data = defaultdict(list)

        for subfolder in self.subfolders:
            for metric_name in self.metric_names:
                with open(os.path.join(self.folder, subfolder, metric_name + ".json")) as f:
                    data[metric_name].append(
                        pd.DataFrame(json.loads(f.read()), index=[0])
                    )

        means = {}

        for metric_name in self.metric_names:
            means[metric_name] = pd.concat(data[metric_name]).groupby(level=0).mean().to_dict(orient="records")[0]

        if len(means) == 1:
            means = means[list(means.keys())[0]]

        return means

