import json
import torch

from tqdm import tqdm

from .inference import InferenceMethod
from cap_utils import plot_and_save_model_results
from utils.torch_utils.multiio import *

import models
import cap_utils as lu



class CoverInference(InferenceMethod):
    def __init__(self):
        super().__init__()

    def __call__(self, model, device):
        return super().__call__(model)


class MultipartCoverPredictionInference(CoverInference):
    """Aggregates multiple inferences into one.
    This is useful for models that have multiple outputs, such as cover and phenology prediction models."""
    def __init__(self, inferences, data_loader, model_kwargs):
        """
        Args:
            inferences (list): List of CoverPredictionInference objects to aggregate.
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be used for inference.
            model_kwargs (dict): Additional keyword arguments to be passed to the model during inference.
        """
        super().__init__()

        self.inferences = inferences
        self.data_loader = data_loader
        self.model_kwargs = model_kwargs

    def __call__(self, model, device):
        model = model.to(device)

        predictions = {inference.name: [] for inference in self.inferences}

        model.eval()

        print(f"Inferencing model:")
        print(json.dumps(self.model_kwargs, indent=True))

        for i, t in enumerate(tqdm(self.data_loader)):
            x, _, _ = _unpack_tuple(t, device)

            with torch.amp.autocast(device.type), torch.no_grad():
                pred = _predict(
                    x,
                    model,
                    **self.model_kwargs,
                )

                x = None
                torch.cuda.empty_cache()

            for inference in self.inferences:
                pred_inf = inference.select_prediction(pred)

                for prediction in pred_inf:
                    predictions[inference.name].append(prediction.cpu().numpy())

        for inference in self.inferences:
            inference.log_predictions(predictions[inference.name])


class CoverPredictionInference(CoverInference):
    """
    Inference method for cover prediction models.
    This class handles the inference process, including loading data, making predictions, and saving results.
    """
    def __init__(self, name, data_loader, split_name, verbose=False, predictions_folder=None, output_index=0, output_key=None, multivalue_names=None, avg_modes=None, model_kwargs=None):
        """
        Args:
            name (str): Name/tag for the inference run.
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to be used for inference.
            split_name (str): Name of the dataset split (e.g., "train", "val", "cvX").
            verbose (bool): Whether to print verbose output.
            predictions_folder (str): Folder to save the predictions.
            output_index (int): Index of the output to be extracted during inference.
            output_key (str): Key of the output to be used for inference, if prediction is a dictionary.
            multivalue_names (dict): Dictionary of multivalue names to give names to each dimension of the output, if there are several. E.g., {0: "flowering", 1: "senescence"}.
            avg_modes (dict): Dictionary of average modes for the outputs.
            model_kwargs (dict): Additional keyword arguments to be passed to the model during inference.
        """
        super().__init__()
        self.name = name
        self.data_loader = data_loader
        self.split_name = split_name
        self.verbose = verbose

        self.multivalue_names = multivalue_names
        self.output_index = output_index
        self.output_key = output_key
        self.predictions_folder = predictions_folder
        self.avg_modes = avg_modes
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

    def __call__(self, model, device):
        model = model.to(device)

        predictions = []

        model.eval()

        print(f"Inferencing model:")
        print(json.dumps(self.model_kwargs, indent=True))

        for i, t in enumerate(tqdm(self.data_loader)):
            x, _, _ = _unpack_tuple(t, device)

            with torch.amp.autocast(device.type), torch.no_grad():
                pred = _predict(
                    x,
                    model,
                    **self.model_kwargs,
                )

                x = None
                torch.cuda.empty_cache()

            pred = self.select_prediction(pred)

            for prediction in pred:
                predictions.append(prediction.cpu().numpy())

        self.log_predictions(predictions)

    def select_prediction(self, pred):
        if isinstance(pred, (tuple, list)):
            pred = pred[self.output_index]

        if self.output_key is not None:
            pred = pred[self.output_key]

        return pred


    def log_predictions(self, predictions):
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
                    include_targets=False,
                )


class TimeSeriesCoverPredictionInference(CoverPredictionInference):
    """
    Inference method for cover prediction models with time series prediction.
    This class handles the inference process, including loading data, making predictions, and saving results.
    """
    def __init__(self, *args, ts_prediction_model, training_data_loader=None, train_ts_model="auto", intermediate_avg_key=None, n_supplementary_time_steps=0, **kwargs):
        """
        Args:
            ts_prediction_model (object): Time series prediction model to be used for inference.
            training_data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset, if applicable.
            train_ts_model (bool): Whether to train the time series model. Default is "auto", which means it will be trained if required.
            intermediate_avg_key (str): Key for averaging intermediate results (e.g. "hour"), e.g. hourly data can be aggregated into a daily average, if the time series model is based on daily data.
            n_supplementary_time_steps (int): Number of supplementary time steps to consider during inference. Only relevant for time step cover prediction models.
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
                model, device, self.training_data_loader, apply_ts_model=False, group_outputs=True, filter_gt=False,
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
            model, device, self.data_loader, apply_ts_model=True, filter_gt=True,
        )

        self.log_pi(predictions, targets, indices)

    def _predict(self, model, device, data_loader, apply_ts_model=False, group_outputs=False, filter_gt=False, detach_predictions=True):
        model = model.to(device)

        predictions = []
        targets = []
        indices = []

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

            for ts_idx, t in enumerate(zip(ts_x, ts_y)):
                x, _, _ = _unpack_tuple(t, device)

                if not isinstance(self.model_kwargs, (tuple, list)):
                    model_kwargs = [self.model_kwargs]
                else:
                    model_kwargs = self.model_kwargs

                with torch.amp.autocast(device.type), torch.no_grad():
                    if self.n_supplementary_time_steps > 0:
                        supplementary_ts = ts_predictions[-1:-1 - self.n_supplementary_time_steps:-1]

                        if len(supplementary_ts) > 0:
                            if len(supplementary_ts) == 1:
                                supplementary_ts = torch.tensor(supplementary_ts[0])
                            else:
                                supplementary_ts = torch.stack(supplementary_ts, model.temporal_dim)

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

                # Here, we have a list of prediction items:
                # [shape(b, kwarg_output1), shape(b, kwarg_output2), ...]

                # Interpret all batch items as time series elements and append to ts list
                for pt in zip(*pred):
                    pt = [pt_component.cpu().numpy() for pt_component in pt]
                    ts_predictions.append(pt)

            if apply_ts_model and self.ts_prediction_model is not None:
                # Reformat ts_predictions to fit time series format
                ts_predictions = tuple(list(v) for v in zip(*ts_predictions))

                print("Eval seq ids:", data_loader.dataset.indices_to_seq_ids(ts_indices))
                ts_predictions = self.ts_prediction_model(
                    ts_predictions,
                    seq_ids=data_loader.dataset.indices_to_seq_ids(ts_indices),
                    clip=(0.0, 1.0),
                )
            else:
                # Only the first prediction component is a valid prediction, if no time-series model is applied
                ts_predictions = [p[0] for p in ts_predictions]

            if filter_gt:
                print("Before filtering:", len(ts_predictions), len(ts_indices))

                # Filter according to gt
                ts_predictions = [p for p, is_gt in zip(ts_predictions, ts_data_is_ground_truth) if is_gt]
                ts_indices = [vidx for vidx, is_gt in zip(ts_indices, ts_data_is_ground_truth) if is_gt]

                print("After filtering:", len(ts_predictions), len(ts_indices))
            else:
                print("Data length:", len(ts_predictions), len(ts_indices))

            if self.intermediate_avg_key:
                ts_predictions, _, ts_indices = lu.average_over_index_key(
                    self.intermediate_avg_key,
                    predictions=ts_predictions,
                    targets=None,
                    indices=ts_indices,
                    id_headers=data_loader.dataset.index_names,
                )

            if group_outputs:
                predictions.append(ts_predictions)
                indices.append(ts_indices)
            else:
                for prediction in ts_predictions:
                    predictions.append(prediction)
                for index in ts_indices:
                    indices.append(index)

        print(f"{len(predictions)} final predictions")

        return predictions, indices

    def log_pi(self, predictions, indices):
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

                lu.pi_to_file(
                    predictions=predictions,
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

                        lu.pi_to_file(
                            predictions=predictions,
                            indices=indices,
                            headers=headers,
                            id_headers=id_headers,
                            folder=self.predictions_folder,
                            file_prefix=self.split_name + prefix + "_" + avg_prefix,
                            output_index=self.output_index,
                            multivalue_index=mv_idx,
                            avg_key=v,
                        )


class CoverSegmentationInference(InferenceMethod):
    """Inference method for cover segmentation models. This class handles the inference process, including loading data, making predictions, and saving results.
    """
    def __init__(self, transforms, save_path, split=None, images=None, classes=None, output_index=0, model_kwargs=None):
        """
        Args:
            transforms (callable): Transformations to be applied to the images.
            save_path (str): Path to save the segmentation results.
            split (str): Name of the dataset split (e.g., "train", "val", "cvX"). Only for designating the save path. Can be None.
            images (list): List of images to be used for inference.
            classes (list): List of class names for the segmentation task.
            output_index (int): Index of the output to be extracted during inference.
            model_kwargs (dict): Additional keyword arguments to be passed to the model during inference.
        """
        super().__init__()

        self.transforms = transforms
        self.save_path = save_path
        self.split = split
        self.output_index = output_index
        self.images = images
        self.classes = classes

        if model_kwargs is None:
            self.model_kwargs = {}
        else:
            self.model_kwargs = model_kwargs

    def __call__(self, model, device):
        print("Saving segmentation images to", self.save_path)

        plot_and_save_model_results(
            model,
            image_transform=self.transforms,
            images=self.images,
            save_path=self.save_path,
            split=self.split,
            device=device,
            output_index=self.output_index,
            species_list=self.classes,
            **self.model_kwargs,
        )