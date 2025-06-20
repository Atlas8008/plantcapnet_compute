import os
import re
import torch
import torchmetrics

from copy import deepcopy
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

import losses.plant_cover as pcl
from losses.plant_cover import ExponentialAbsoluteErrorLoss, MeanScaledAbsoluteErrorLoss, MeanSquaredNormalizedErrorLoss
import losses.plant_cover as PCL
import losses.segmentation as SL
from losses.util import CombinedLoss
from metrics.plant_cover import MeanScaledAbsoluteError, MeanNormalizedAbsoluteError, MeanAbsoluteErrorWClassMask
from metrics.ancova import AncovaR2
from models import time_series

from data import StandardizedPlantCommunityDataset

from utils import torch_utils
from utils import augmentations as augs
from utils.utils import str2bool

from multiprocessing import cpu_count
from evaluation.cover_prediction import CoverPredictionEvaluation, CoverSegmentationEvaluation, TimeSeriesCoverPredictionEvaluation
from inference.cover_prediction import CoverSegmentationInference
from training_configurations.configuration import Configuration
from models.spt_model import EnsembleModel
from cap_utils import augmentation
from .meta_configuration import GeneralCoverPredictionMetaConfiguration


LOSSES = {
    "l1": torch.nn.L1Loss,
    "mae": torch.nn.L1Loss,
    "mse": torch.nn.MSELoss,
    "rmse": pcl.RMSELoss,
    "mp3e": lambda *args, **kwargs: pcl.MPNELoss(*args, n=3, *kwargs),
    "mp4e": lambda *args, **kwargs: pcl.MPNELoss(*args, n=4, *kwargs),
    "mp5e": lambda *args, **kwargs: pcl.MPNELoss(*args, n=5, *kwargs),
}

WEIGHTED_LOSSES = {
    "l1": pcl.WeightedL1Loss,
    "mae": pcl.WeightedL1Loss,
    "mse": pcl.WeightedMSELoss,
    "rmse": pcl.WeightedRMSELoss,
}


def _get_loss(losses, dataset_means=None, dataset_stds=None):
    criterions = []
    weights = []

    weight_loss_parser = re.compile("^([0-9.e-]*[0-9])(.*)$")

    for loss in losses.split(","):
        if match := weight_loss_parser.match(loss):
            weight = float(match.group(1))
            loss = match.group(2)
        else:
            weight = 1.0

        weights.append(weight)

        if loss == "mae" or loss == "l1":
            print("Using l1 loss")
            criterion = nn.L1Loss()
        elif loss == "mnae":
            print("Using MNAE loss")
            criterion = MeanScaledAbsoluteErrorLoss(
                means=dataset_means,
                stds=dataset_stds
            )
        elif loss == "meae":
            print("Using MEAE loss")
            criterion = ExponentialAbsoluteErrorLoss()
        elif loss == "mse": # MeanSquaredErrorLoss
            print("Using MSE loss")
            criterion = nn.MSELoss()
        elif loss == "msne": # MeanSquaredNormalizedErrorLoss
            print("Using MSNE loss")
            criterion = MeanSquaredNormalizedErrorLoss(
                means=dataset_means,
                stds=dataset_stds,
            )
        elif loss == "bce":
            criterion = PCL.PCBCE()#nn.BCELoss()
        elif loss == "sidxmae":
            criterion = PCL.ShannonIndexMAELoss()
        elif loss == "sidxmse":
            criterion = PCL.ShannonIndexMSELoss()
        elif loss == "sidxrmse":
            criterion = PCL.ShannonIndexRMSELoss()
        elif loss == "kld":
            criterion = PCL.KLDLoss()
        elif loss == "skld":
            criterion = PCL.KLDLoss(smoothing_value=1.0)
        elif loss == "jsd":
            criterion = PCL.JensenShannonDivergence()
        elif loss == "sjsd":
            criterion = PCL.JensenShannonDivergence(smoothing_value=1.0)
        elif loss == "dice":
            criterion = SL.DiceLoss()
        elif loss == "bhat":
            criterion = PCL.BhattacharyyaLoss()
        elif loss == "bhat2":
            criterion = PCL.BhattacharyyaLoss(smoothing_value=0.0, epsilon=1)
        elif loss == "bhatc":
            criterion = PCL.BhattacharyyaLoss(smoothing_value=0.0, epsilon=0, coefficient_only=True)
        elif loss == "hell":
            criterion = PCL.HellingerLoss()
        else:
            raise ValueError("Invalid loss function: " + loss)

        criterions.append(criterion)

    if len(criterions) > 1:
        criterion = CombinedLoss(criterions, weights)
    else:
        criterion = criterions[0]

    return criterion


class CoverPredictionMetaConfiguration(GeneralCoverPredictionMetaConfiguration):
    def __init__(self, experiment_name, args, eval_args, additional_info=None):
        super().__init__(experiment_name, args, eval_args, additional_info)

        self.logger_tags = ["default"]

        self.task_name = "cover"

    def _init_subconfigs(self):
        configs = [
            CoverPredictionConfiguration(
                self.experiment_name,
                self.args,
                self.eval_args,
                split=split,
                dataset_class=self.dataset_class,
            ) for split in self.splits
        ]

        for config in configs:
            config.add_info(**self._additional_info)

        return configs



class CoverPredictionConfiguration(Configuration):
    def __init__(self, experiment_name, args, eval_args, dataset_class, split="val"):
        super().__init__(experiment_name, args)

        self.eval_args = eval_args

        print(args.image_size)
        assert len(args.image_size) <= 2, "Too many arguments for image size, maximum is 2."

        if len(args.image_size) == 2:
            self._image_size = tuple(args.image_size)
        else:
            self._image_size = (args.image_size[0] // 2, args.image_size[0])
        self._image_size_highres = (1536, 2816)

        self.split = split
        self.dataset_class = dataset_class

        if self.args.dataset.lower() == "none":
            print("No dataset specified, using dummy dataset")
            self._n_classes = 1
        elif self.dataset_class.class_names is not None:
            self._n_classes = len(self.dataset_class.class_names)
        else:
            self._n_classes = len(StandardizedPlantCommunityDataset.get_class_names(self.args.dataset))

        self.task_name = "cover"
        self.add_meta_info(split=self.split)

        self.include_phenology = self.args.pheno_model not in (None, "none")

        if hasattr(self.args, "use_combined_model") and self.args.use_combined_model:
            print("Using combined model")
            self.segm_model_kwargs = {
                "seg_model_kwargs": {
                    "use_deocclusion": False,
                    "segmentation_kwargs": {
                        "mode": "cam",
                    }
                }
            }
        else:
            self.segm_model_kwargs = {}

        self.num_workers = min(cpu_count(), self.args.max_workers)

    @property
    def n_model_outputs(self):
        return self._n_classes

    @property
    def training_transforms(self):
        start_augs, mid_augs, final_augs = augmentation.get_augmentations(
            self.args.aug_strat, image_size=self._image_size)

        return transforms.Compose(start_augs + [
            transforms.Resize(self._image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        ] + mid_augs + [
            transforms.ToTensor(),
            augs.Preprocess(self.args.normalization),
        ] + final_augs)

    @property
    def eval_transforms(self):
        return transforms.Compose([
            transforms.Resize(self._image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            augs.Preprocess(self.args.normalization),
        ])

    @property
    def loss(self):
        losses = self.args.loss

        criterion = _get_loss(losses, self.training_dataset.dataset_means, self.training_dataset.dataset_stds)

        print("Plant Cover Prediction Loss:", criterion)

        if self.include_phenology:
            criterion2 = _get_loss(self.args.pheno_loss)

            criterion = [criterion, criterion2]

        print("Loss is ", criterion)

        return criterion

    @property
    @Configuration.cache
    def training_dataset(self):
        if StandardizedPlantCommunityDataset.is_dataset(self.args.dataset):
            ds_name = self.args.dataset.split(",")[0]

            ds_params = self.args.dataset.split(",")[1:]
            ds_params = {k: v for k, v in [p.split("=") for p in ds_params]}

            apply_masks = str2bool(ds_params.get("masks", "False"))

            return StandardizedPlantCommunityDataset(
                folder_name=ds_name,
                split="train",
                mode=self.args.dataset_mode,
                apply_masks=apply_masks,
                transform=self.training_transforms,
                pheno_model=self.args.pheno_model,
            )
        else:
            raise ValueError("Invalid dataset name: " + self.args.dataset)

    @property
    @Configuration.cache
    def eval_dataset(self):
        if StandardizedPlantCommunityDataset.is_dataset(self.args.dataset):
            ds_name = self.args.dataset.split(",")[0]

            ds_params = self.args.dataset.split(",")[1:]
            ds_params = {k: v for k, v in [p.split("=") for p in ds_params]}

            apply_masks = str2bool(ds_params.get("masks", "False"))

            return StandardizedPlantCommunityDataset(
                folder_name=ds_name,
                split="validation",
                mode=self.args.dataset_mode,
                apply_masks=apply_masks,
                transform=self.eval_transforms,
                pheno_model=self.args.pheno_model,
            )
        else:
            raise ValueError("Invalid dataset name: " + self.args.dataset)

    @property
    def training_loader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    @property
    def eval_loader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def model_kwargs(self):
        return {
            "head": "cover_prediction",
            "mc_params": self.args.mc_params,
            **self.segm_model_kwargs,
        }

    @property
    def metrics(self):
        # Cover metrics
        metrics = [
            torchmetrics.MeanAbsoluteError(),
            MeanScaledAbsoluteError(self.training_dataset.dataset_means),
            MeanNormalizedAbsoluteError(
                means=self.training_dataset.dataset_means,
                stds=self.training_dataset.dataset_stds
            ),
            AncovaR2(
                mean_over_last_dim=None,
                name="AncovaR2(C)",
            ),
        ]

        if hasattr(self.training_dataset, "class_is_frequent"):
            metrics.append(
                AncovaR2(
                    mean_over_last_dim=None,
                    name="AncovaR2(C_freq)",
                    class_mask=self.training_dataset.class_is_frequent,
                )
            )
            metrics.append(
                AncovaR2(
                    mean_over_last_dim=None,
                    name="AncovaR2(C_infreq)",
                    class_mask=[
                        v * -1 + 1 for v in self.training_dataset.class_is_frequent
                    ],
                )
            )


        if self.include_phenology:
            metrics = [metrics, [
                MeanAbsoluteErrorWClassMask(
                    class_mask=self.training_dataset.class_has_phenology,
                ),
                AncovaR2(
                    mean_over_last_dim=False,
                    class_mask=self.training_dataset.class_has_phenology
                ),
                AncovaR2(
                    mean_over_last_dim=False,
                    class_mask=self.training_dataset.class_has_phenology,
                    last_dim_idx=0,
                    name="AncovaR2(F)",
                ),
                AncovaR2(
                    mean_over_last_dim=False,
                    class_mask=self.training_dataset.class_has_phenology,
                    last_dim_idx=1,
                    name="AncovaR2(S)",
                ),
            ]]

        return metrics

    @property
    def epochs(self):
        return self.args.n_epochs

    @property
    def evaluation_methods(self):
        cover_prediction_kwargs = {
            "head": "cover_prediction",
            "mode": "cover_prediction",
            **self.segm_model_kwargs,
        }
        segmentation_kwargs = {
            "head": "cover_prediction",
            "mode": "segmentation",
            **self.segm_model_kwargs,
        }

        if self.eval_args.ts_include_feats:
            ts_cover_prediction_kwargs_lst = [[
                cover_prediction_kwargs,
                {
                    "head": "mean_features",
                },
            ]]
        else:
            ts_cover_prediction_kwargs_lst = [cover_prediction_kwargs]

        return self._get_evaluation_methods(
            inference_configs=["default"],
            cover_prediction_kwargs_lst=[cover_prediction_kwargs],
            segmentation_kwargs_lst=[segmentation_kwargs],
            ts_cover_prediction_kwargs_lst=ts_cover_prediction_kwargs_lst,
        )

    def _get_evaluation_methods(self, inference_configs, cover_prediction_kwargs_lst, segmentation_kwargs_lst, ts_cover_prediction_kwargs_lst=None):

        logger_tags = []
        eval_methods = []

        evaluation_settings = self.dataset_class.EVALUATION_SETTINGS

        if ts_cover_prediction_kwargs_lst is None:
            ts_cover_prediction_kwargs_lst = cover_prediction_kwargs_lst

        for inference_config, cover_prediction_kwargs, segmentation_kwargs, ts_cover_prediction_kwargs in zip(
                inference_configs,
                cover_prediction_kwargs_lst,
                segmentation_kwargs_lst,
                ts_cover_prediction_kwargs_lst,
            ):
            cover_metrics = self.metrics
            avg_modes = evaluation_settings.average_modes

            if self.include_phenology:
                pheno_eval = CoverPredictionEvaluation(
                    name="phenology_" + inference_config,
                    data_loader=self.eval_loader,
                    metrics=self.metrics[1],
                    split_name=self.split,
                    verbose=True,
                    logger_tags=logger_tags,
                    predictions_folder=os.path.join(
                        self.output_folder,
                        "data",
                        self.experiment_name,
                        inference_config
                    ),
                    output_index=1,
                    multivalue_names=self.eval_dataset.pheno_value_names,
                    model_kwargs=cover_prediction_kwargs,
                    avg_modes=avg_modes,
                )
                cover_metrics = self.metrics[0]

                eval_methods.append(pheno_eval)

            pred_eval = CoverPredictionEvaluation(
                name="cover_prediction_" + inference_config,
                data_loader=self.eval_loader,
                metrics=cover_metrics,
                split_name=self.split,
                verbose=True,
                logger_tags=logger_tags,
                predictions_folder=os.path.join(
                    self.output_folder,
                    "data",
                    self.experiment_name,
                    inference_config
                ),
                output_index=0,
                model_kwargs=cover_prediction_kwargs,
                avg_modes=avg_modes,
            )
            eval_methods.append(pred_eval)

            if hasattr(self, "eval_args") and self.eval_args.ts_eval:
                ts_eval_loader = DataLoader(
                    self.eval_dataset.as_time_series(
                        data_kind=self.eval_args.ts_eval_data_kind,
                    ),
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                ts_train_loader = DataLoader(
                    self.training_dataset.as_time_series(
                        data_kind=self.eval_args.ts_train_data_kind,
                        transform=ts_eval_loader.dataset.transform,
                        target_transform=ts_eval_loader.dataset.target_transform,
                    ),
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

                intermediate_avg_keys = evaluation_settings.ts_intermediate_average_keys

                pred_eval = [TimeSeriesCoverPredictionEvaluation(
                    name="cover_prediction_ts_" + inference_config + ("_inter" + iak if iak else ""),
                    data_loader=ts_eval_loader,
                    metrics=cover_metrics,
                    split_name=self.split,
                    verbose=True,
                    logger_tags=logger_tags,
                    predictions_folder=os.path.join(
                        self.output_folder,
                        "data",
                        self.experiment_name,
                        inference_config
                    ),
                    output_index=0,
                    model_kwargs=ts_cover_prediction_kwargs,
                    ts_prediction_model=time_series.get_time_series_model(
                        self.eval_args.ts_model_spec,
                        n_out=ts_eval_loader.dataset.get_num_classes(),
                    ),
                    train_ts_model="auto",
                    training_data_loader=ts_train_loader,
                    intermediate_avg_key=iak,
                ) for iak in intermediate_avg_keys]
                eval_methods.extend(pred_eval)

                if self.include_phenology:
                    pred_eval = [TimeSeriesCoverPredictionEvaluation(
                        name="phenology_ts_" + inference_config + ("_inter" + iak if iak else ""),
                        data_loader=ts_eval_loader,
                        metrics=self.metrics[1],
                        split_name=self.split,
                        verbose=True,
                        logger_tags=logger_tags,
                        predictions_folder=os.path.join(
                            self.output_folder,
                            "data",
                            self.experiment_name,
                            inference_config
                        ),
                        output_index=1,
                        multivalue_names=self.eval_dataset.pheno_value_names,
                        model_kwargs=ts_cover_prediction_kwargs,
                        ts_prediction_model=time_series.get_time_series_model(
                            self.eval_args.ts_model_spec,
                            n_out=ts_eval_loader.dataset.get_num_classes(),
                        ),
                        train_ts_model="auto",
                        training_data_loader=ts_train_loader,
                        intermediate_avg_key=iak,
                    ) for iak in intermediate_avg_keys]
                    eval_methods.extend(pred_eval)

            if evaluation_settings.segmentation_evaluation and self.eval_args.segmentation_eval == "auto" or self.eval_args.segmentation_inference == True:
                segmentation_save_path = None
                error_map_save_path = None

                if self.eval_args.save_eval_segmaps:
                    segmentation_save_path = os.path.join(
                        self.output_folder,
                        "images",
                        self.experiment_name,
                        inference_config,
                        self.split,
                        "eval",
                        "segmaps",
                    )
                if self.eval_args.save_eval_errmaps:
                    error_map_save_path = os.path.join(
                        self.output_folder,
                        "images",
                        self.experiment_name,
                        inference_config,
                        self.split,
                        "eval",
                        "errmaps",
                    )

                seg_eval = CoverSegmentationEvaluation(
                    name="segmentation_" + inference_config,
                    image_size_hw=self._image_size,
                    normalization=self.args.normalization,
                    verbose=True,
                    include_dead_litter=self.args.include_dead_litter,
                    logger_tags=logger_tags,
                    output_index=0,
                    segmentation_save_path=segmentation_save_path,
                    error_map_save_path=error_map_save_path,
                    model_kwargs=segmentation_kwargs,
                )
                eval_methods.append(seg_eval)

        return eval_methods

    @property
    def inference_methods(self):
        tag = "default"

        evaluation_settings = self.dataset_class.EVALUATION_SETTINGS

        if evaluation_settings.segmentation_inference and self.eval_args.segmentation_inference == "auto" or self.eval_args.segmentation_inference == True:

            class_names = self.dataset_class.class_names

            if class_names is None:
                class_names = self.dataset_class.get_class_names(self.args.dataset)

            return [
                CoverSegmentationInference(
                    save_path=os.path.join(
                        self.output_folder,
                        "images",
                        self.experiment_name,
                        tag,
                    ),
                    split=self.split,
                    transforms=self.eval_transforms,
                    output_index=0,
                    classes=class_names,
                    images=evaluation_settings.segmentation_inference_images,
                    model_kwargs={
                        "head": "cover_prediction",
                        "mode": "segmentation",
                        **self.segm_model_kwargs,
                    }
                )
            ]
        else:
            return []

    @property
    def metric_path(self):
        return os.path.join(self.metric_folder, self.experiment_name, self.split)

    def set_model_trainability(self, model):
        torch_utils.set_trainability(
            model,
            True,
        )
        torch_utils.set_trainability(
            model.feature_extractor.base_model,
            self.args.train_base_model,
        )

    def get_optimizer(self, model):
        if isinstance(self.args.learning_rate, list) and len(self.args.learning_rate) >= 2:
            params_non_pheno = (param for name, param in model.named_parameters() if "phenology_module" not in name)
            params_pheno = (param for name, param in model.named_parameters() if "phenology_module" in name)

            params_non_pheno = list(params_non_pheno)
            params_pheno = list(params_pheno)

            print("Non pheno:", len(params_non_pheno))
            print("Pheno:", len(params_pheno))

            return torch.optim.AdamW(
                params=[
                    {"params": params_non_pheno},
                    {"params": params_pheno, "lr": self.args.learning_rate[1]},
                ],
                lr=self.args.learning_rate[0],
                weight_decay=self.args.weight_decay,
                eps=1e-4,
            )
        else:
            if isinstance(self.args.learning_rate, list):
                learning_rate = self.args.learning_rate[0]
            else:
                learning_rate = self.args.learning_rate

        return torch.optim.AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-4,
        )

    def get_scheduler(self, optimizer):
        print(self.args)
        if self.args.lr_scheduler in ("default", "step"):
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                [
                    int(self.args.n_epochs * 0.5),
                    int(self.args.n_epochs * 0.75),
                ]
            )
        elif match := re.match("cos([0-9]+),([0-9.]+)", self.args.lr_scheduler):
            t0 = int(match.group(1))
            t_mult = int(match.group(2))
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t0,
                T_mult=t_mult,
            )


    @property
    def keep_saved_model(self):
        return self.args.keep_saved_model

    def modify_model_with_configs(self, model, config_dict, device):
        if self.args.use_combined_model:
            model.additional_segmentation_module = torch.load(
                config_dict["segmentation"].model_save_path,
                map_location=device,
                weights_only=False,
            )
            torch_utils.set_trainability(
                model.additional_segmentation_module,
                False
            )
        else:
            model.additional_segmentation_module = None

        return model

    def restore_model_from_checkpoint(self, model):
        print("Restoring model from", self.model_save_path)
        m = torch.load(self.model_save_path, map_location="cpu", weights_only=False)

        model.feature_extractor.load_state_dict(
            m.feature_extractor.state_dict()
        )
        model.segmentation_module.load_state_dict(
            m.segmentation_module.state_dict()
        )
        model.cover_module.load_state_dict(
            m.cover_module.state_dict()
        )

        print("Restored model hash:", hash(str(list(model.parameters()))))

    def maybe_build_ensemble(self, model, config_dict, device):
        models = []

        if self.args.ensemble_model_names is not None:
            if not self.args.ensemble_meta_tag_suffixes or self.args.ensemble_meta_tag_suffixes in (None, "none"):
                meta_tag_suffixes = [""]
            else:
                meta_tag_suffixes = self.args.ensemble_meta_tag_suffixes

            for model_name in self.args.ensemble_model_names:
                for meta_tag_suffix in meta_tag_suffixes:
                    model_path = f"meta/model_links/{self.task_name}/{model_name}{self.meta_info_str}{meta_tag_suffix}.pth"

                    print("Loading model from", model_path)

                    models.append(
                        torch.load(
                            model_path,
                            map_location="cpu",
                            weights_only=False,
                        )
                    )
        else:
            available_ensemble_lists = 3

            print("Ensemble model:")
            ensemble_tags = self.args.ensemble_tags
            ensemble_epochs = self.args.ensemble_epochs
            meta_tag_suffixes = self.args.ensemble_meta_tag_suffixes

            if not ensemble_tags or ensemble_tags in (None, "none"):
                ensemble_tags = [None]
                available_ensemble_lists -= 1
            if not ensemble_epochs or ensemble_epochs in (None, "none"):
                ensemble_epochs = [None]
                available_ensemble_lists -= 1
            if not meta_tag_suffixes or meta_tag_suffixes in (None, "none"):
                meta_tag_suffixes = [None]
                available_ensemble_lists -= 1

            print("Available:", available_ensemble_lists)
            print(ensemble_epochs)
            # No ensemble to be built --> return normal model
            if available_ensemble_lists == 0:
                return model, False

            for ensemble_tag in ensemble_tags:
                for meta_tag_suffix in meta_tag_suffixes:
                    for epoch_count in ensemble_epochs:
                        config = deepcopy(self)

                        if ensemble_tag is not None:
                            config.args.tag = ensemble_tag
                        if epoch_count is not None:
                            config.args.n_epochs = epoch_count
                        if meta_tag_suffix is not None:
                            config._meta_info["tag"] = self._meta_info["tag"] + meta_tag_suffix
                            orig_check_point, ext = os.path.splitext(self._additional_info["restore_checkpoint"])
                            config._additional_info["restore_checkpoint"] = orig_check_point + meta_tag_suffix + ext

                        config.args.ensemble_tags = None
                        config.args.ensemble_epochs = None
                        config.args.ensemble_meta_tag_suffixes = None

                        models.append(
                            torch.load(
                                config.model_save_path,
                                map_location="cpu",
                                weights_only=False,
                            )
                        )
                        print("Loaded model from", config.model_save_path)

        ensemble_model = EnsembleModel(
            *models,
            dynamic_device=device,
        )
        print("Built ensemble model from", len(models), "models")

        return ensemble_model, True