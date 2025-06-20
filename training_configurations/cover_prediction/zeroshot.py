import os
import torch

from copy import deepcopy

from utils import torch_utils
from models.spt_model import EnsembleModel

from inference.cover_prediction import CoverSegmentationInference
from .cover_prediction import GeneralCoverPredictionMetaConfiguration, CoverPredictionConfiguration


class ZeroShotCoverPredictionMetaConfiguration(GeneralCoverPredictionMetaConfiguration):
    def __init__(self, experiment_name, args, eval_args, additional_info=None) -> None:
        super().__init__(experiment_name, args, eval_args, additional_info)

        self.logger_tags = args.inference_configurations

        print("Logger tags:", self.logger_tags)

        self.task_name = "zeroshot"

    def _init_subconfigs(self):
        configs = [
            ZeroShotCoverPredictionConfiguration(
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



class ZeroShotCoverPredictionConfiguration(CoverPredictionConfiguration):
    def __init__(self, experiment_name, args, eval_args, dataset_class, split="val"):
        super().__init__(experiment_name, args, eval_args, dataset_class, split)

        self.task_name = "zeroshot"
        self.segm_model_kwargs = {}

    @property
    def epochs(self):
        return 0

    @property
    def model_kwargs(self):
        return {
            "head": "zeroshot_cover_prediction",
            "mode": "cover_prediction",
            "submodel_kwargs": {
                "head": "segmentation",
                "use_deocclusion": self.args.use_deocclusion_module,
                "segmentation_kwargs": {
                    "mode": "cam",
                },
                "deocclusion_kwargs": {
                    "mode": "cam",
                }
            }
        }

    @property
    def training_transforms(self):
        return None

    def set_model_trainability(self, model):
        torch_utils.set_trainability(
            model,
            False,
        )

    def __resolve_inference_config(self, inference_config):
        d = {}

        arg_list = inference_config.split(",")

        for arg in arg_list:
            if arg[0] == "b":
                d["bg_type"] = arg[1:]
            elif arg[0] == "d":
                d["discretization"] = arg[1:]
            else:
                raise ValueError("Invalid inference arg: " + arg)

        return d

    @property
    def evaluation_methods(self):
        cover_prediction_kwargs_lst = [{
                    "head": "zeroshot_cover_prediction",
                    "mode": "cover_prediction",
                    "enriched_prediction": self.args.enriched_eval,
                    "interlaced_prediction": self.args.enriched_eval,
                    "submodel_kwargs": {
                        "head": "segmentation",
                        "use_deocclusion": self.args.use_deocclusion_module,
                        "segmentation_kwargs": {
                            "mode": "cam",
                        },
                        "deocclusion_kwargs": {
                            "mode": "cam",
                        },
                    },
                **self.__resolve_inference_config(inference_config),
            } for inference_config in self.args.inference_configurations
        ]
        segmentation_kwargs_lst = [{
                "head": "zeroshot_cover_prediction",
                "mode": "segmentation",
                "enriched_prediction": self.args.enriched_eval,
                "interlaced_prediction": self.args.enriched_eval,
                "submodel_kwargs": {
                    "head": "segmentation",
                    "use_deocclusion": self.args.use_deocclusion_module,
                    "segmentation_kwargs": {
                        "mode": "cam",
                    },
                    "deocclusion_kwargs": {
                        "mode": "cam",
                    },
                },
                **self.__resolve_inference_config(inference_config),
            } for inference_config in self.args.inference_configurations
        ]

        if self.eval_args.ts_include_feats:
            ts_cover_prediction_kwargs_lst = [[
                cover_prediction_kwargs,
                {
                    "head": "mean_features",
                },
            ] for cover_prediction_kwargs in cover_prediction_kwargs_lst]
        else:
            ts_cover_prediction_kwargs_lst = cover_prediction_kwargs_lst

        return self._get_evaluation_methods(
            inference_configs=self.args.inference_configurations,
            cover_prediction_kwargs_lst=cover_prediction_kwargs_lst,
            segmentation_kwargs_lst=segmentation_kwargs_lst,
            ts_cover_prediction_kwargs_lst=ts_cover_prediction_kwargs_lst,
        )


    @property
    def inference_methods(self):
        inferences = []

        evaluation_settings = self.dataset_class.EVALUATION_SETTINGS

        if evaluation_settings.segmentation_inference:
            for inference_config in self.args.inference_configurations:
                inference_args = self.__resolve_inference_config(inference_config)

                tag = inference_config

                inferences.append(
                    CoverSegmentationInference(
                        save_path=os.path.join(
                            self.output_folder,
                            "images",
                            self.experiment_name,
                            tag,
                        ),
                        split=self.split,
                        transforms=self.eval_transforms,
                        classes=self.dataset_class.class_names,
                        images=evaluation_settings.segmentation_inference_images,
                        model_kwargs={
                            "head": "zeroshot_cover_prediction",
                            "mode": "segmentation",
                            "enriched_prediction": self.args.enriched_eval,
                            "interlaced_prediction": self.args.enriched_eval,
                            "enrichment_params": self.args.enrichment_params,
                            "interlace_size": self.args.interlace_size,
                            "submodel_kwargs": {
                                "head": "segmentation",
                                "use_deocclusion": self.args.use_deocclusion_module,
                                "segmentation_kwargs": {
                                    "mode": "cam",
                                },
                                "deocclusion_kwargs": {
                                    "mode": "cam",
                                },
                            },
                            **inference_args,
                        }
                    )
                )

        return inferences

    def modify_model_with_configs(self, model, config_dict, device):
        return model

    def maybe_build_ensemble(self, model, config_dict, device):
        models = []

        # Check, if model names are provided directly, if not, try to construct via parameter changes
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
            meta_tag_suffixes = self.args.ensemble_meta_tag_suffixes
            model_set = self.args.ensemble_model_set

            #ensemble_tags = None

            if not ensemble_tags or ensemble_tags in (None, "none"):
                ensemble_tags = [None]
                available_ensemble_lists -= 1

            if not meta_tag_suffixes or meta_tag_suffixes in (None, "none"):
                meta_tag_suffixes = [None]
                available_ensemble_lists -= 1

            if model_set in (None, "none"):
                model_hashes = [None]
                available_ensemble_lists -= 1
            else:
                model_hashes = ZEROSHOT_ENSEMBLE_BASE_MODEL_HASHES[model_set]

            print("Available:", available_ensemble_lists)
            # No ensemble to be built --> return normal model
            if available_ensemble_lists == 0:
                return model, False

            for meta_tag_suffix in meta_tag_suffixes:
                for ensemble_tag in ensemble_tags:
                    for root_model_hash in model_hashes:
                        config = deepcopy(self)

                        if ensemble_tag is not None:
                            config.args.tag = ensemble_tag
                        if root_model_hash is not None:
                            # Checkpoint name consists of /<path>/<hash><main_tag + meta_tag>.<ext>
                            pth, fname = os.path.split(config._additional_info["restore_checkpoint"])
                            ext = os.path.splitext(fname)[1]
                            # Inject new hash by constructing new model path
                            restore_checkpoint = os.path.join(
                                pth,
                                root_model_hash + self._meta_info["tag"] + ext,
                            )
                            config._additional_info["restore_checkpoint"] = restore_checkpoint

                        if meta_tag_suffix is not None:
                            config._meta_info["tag"] = self._meta_info["tag"] + meta_tag_suffix
                            # Checkpoint name consists of /<path>/<hash><main_tag + meta_tag>.<ext>
                            orig_check_point, ext = os.path.splitext(config._additional_info["restore_checkpoint"])
                            config._additional_info["restore_checkpoint"] = orig_check_point + meta_tag_suffix + ext

                        config.args.ensemble_tags = None
                        config.args.ensemble_meta_tag_suffixes = None
                        #del config.args.ensemble_model_set
                        config.args.ensemble_model_set = "none"

                        param_info_dict = {
                            "parameters": config.param_info,
                            "additional_info": config._additional_info,
                            "meta_info": config._meta_info,
                        }

                        import json
                        print("Ensemble submodel configuration:")
                        print(json.dumps(param_info_dict, indent=True))

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