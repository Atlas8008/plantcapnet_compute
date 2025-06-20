import os

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from multiprocessing import cpu_count

from data import VegetationTimeSeriesDataset, NPrevStepDatasetWrapper

from utils import torch_utils
from utils import augmentations as augs

from training_configurations.configuration import Configuration
from cap_utils import augmentation
from .cover_prediction import GeneralCoverPredictionMetaConfiguration, CoverPredictionConfiguration
from evaluation.cover_prediction import TimeSeriesCoverPredictionEvaluation

NUM_WORKERS = min(cpu_count(), 16)

class SubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        for idx in range(len(self.indices)):
            yield self.indices[idx]


class TemporalCoverPredictionMetaConfiguration(GeneralCoverPredictionMetaConfiguration):
    def __init__(self, base_cover_meta_configuration, experiment_name, args, eval_args, **kwargs):
        args = args.__class__(
            **{
                **vars(base_cover_meta_configuration.args),
                **vars(args)
            }
        )
        self.base_cover_meta_configuration = base_cover_meta_configuration

        super().__init__(experiment_name=experiment_name, args=args, eval_args=eval_args, **kwargs)

        self.logger_tags = ["default"]
        self.task_name = "temporal_cover"

        self.post_evaluation_task_types.append("cover_prediction_tmp")

    def _init_subconfigs(self):
        base_configs = self.base_cover_meta_configuration.configurations

        configs = [
            TemporalCoverPredictionConfiguration(
                base_config,
                experiment_name=self.experiment_name,
                args=self.args,
                eval_args=self.eval_args,
                split=split,
                dataset_class=self.dataset_class,
            ) for split, base_config in zip(self.splits, base_configs)
        ]

        for config in configs:
            config.add_info(**self._additional_info)

        return configs


class TemporalCoverPredictionConfiguration(CoverPredictionConfiguration):
    def __init__(self, base_cover_configuration, args, **kwargs):
        args = args.__class__(
            **{
                **vars(base_cover_configuration.args),
                **vars(args)
            }
        )

        self.base_cover_configuration = base_cover_configuration

        super().__init__(
            args=args,
            **kwargs
        )

        self.task_name = "temporal_cover"

    @property
    def training_transforms(self):
        start_augs, mid_augs, final_augs = augmentation.get_augmentations(
            self.args.aug_strat)

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
    @Configuration.cache
    def training_dataset(self):
        return NPrevStepDatasetWrapper(
            dataset=VegetationTimeSeriesDataset(
                veg_dataset=self.base_cover_configuration.training_dataset,
                transform=self.training_transforms,
            ),
            n=self.args.n_time_steps,
        )

    @property
    @Configuration.cache
    def eval_dataset(self):
        return NPrevStepDatasetWrapper(
            dataset=VegetationTimeSeriesDataset(
                veg_dataset=self.base_cover_configuration.eval_dataset,
                transform=self.eval_transforms,
            ),
            n=self.args.n_time_steps,
        )

    @property
    def training_loader(self):
        indices = []

        for i, ln in enumerate(self.training_dataset.dataset.get_lengths()):
            indices.extend([(i, j) for j in range(ln)])

        return DataLoader(
            self.training_dataset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(indices),
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=True,
        )

    @property
    def eval_loader(self):
        indices = []

        for i, ln in enumerate(self.eval_dataset.dataset.get_lengths()):
            indices.extend([(i, j) for j in range(ln)])

        return DataLoader(
            self.eval_dataset,
            batch_size=1,
            sampler=SubsetSampler(indices),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    @property
    def model_kwargs(self):
        return {
            "head": "temporal_cover_prediction",
            "mc_params": self.args.mc_params,
            **self.segm_model_kwargs,
        }

    @property
    def evaluation_methods(self):
        cover_prediction_kwargs = {
            "head": "temporal_cover_prediction",
            "mode": "cover_prediction",
            **self.segm_model_kwargs,
        }
        segmentation_kwargs = {
            "head": "temporal_cover_prediction",
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

        eval_methods = self._get_evaluation_methods(
            inference_configs=["default"],
            cover_prediction_kwargs_lst=[cover_prediction_kwargs],
            segmentation_kwargs_lst=[segmentation_kwargs],
            ts_cover_prediction_kwargs_lst=ts_cover_prediction_kwargs_lst,
        )

        logger_tags = []
        eval_methods_tmp = []

        evaluation_settings = self.dataset_class.EVALUATION_SETTINGS

        #inference_config = "default"
        inference_config = "default"
        ts_cover_prediction_kwargs = cover_prediction_kwargs

        cover_metrics = self.metrics
        avg_modes = evaluation_settings.average_modes

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

        eval_methods_tmp = [TimeSeriesCoverPredictionEvaluation(
            name="cover_prediction_tmp_" + inference_config + ("_inter" + iak if iak else ""),
            data_loader=ts_eval_loader,
            metrics=cover_metrics,
            split_name=self.split,
            verbose=True,
            logger_tags=logger_tags,
            n_supplementary_time_steps=self.args.n_time_steps,
            predictions_folder=os.path.join(
                self.output_folder,
                "data",
                self.experiment_name,
                inference_config
            ),
            output_index=0,
            model_kwargs=ts_cover_prediction_kwargs,
            ts_prediction_model=None,
            train_ts_model=False,
            training_data_loader=ts_train_loader,
        ) for iak in intermediate_avg_keys]

        return eval_methods + eval_methods_tmp

    def set_model_trainability(self, model):
        torch_utils.set_trainability(
            model,
            True,
        )
        torch_utils.set_trainability(
            model.feature_extractor,
            False,
        )

    def modify_model_with_configs(self, model, config_dict, device):
        model = super().modify_model_with_configs(model, config_dict, device)

        model.trainable_time_steps = self.args.train_time_steps
        model.tmp_share_weights = self.args.tmp_share_weights

        return model