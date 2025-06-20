from copy import deepcopy
import os

from abc import abstractmethod
from data import StandardizedPlantCommunityDataset

from evaluation.bio_eval import BioEvaluation, CorrelationEvaluation
from evaluation.cover_prediction import MetricAggregation
from training_configurations.configuration import MetaConfiguration


join = lambda l: [vi for vo in l for vi in vo]


class GeneralCoverPredictionMetaConfiguration(MetaConfiguration):
    def __init__(self, experiment_name, args, eval_args, additional_info=None) -> None:
        super().__init__(experiment_name, args, additional_info)

        self.eval_args = eval_args
        self.dataset_class = None

        print(self.args)
        print(self.eval_args)

        self.include_phenology = self.args.pheno_model not in (None, "none")
        self.logger_tags = []

        if self.args.dataset.lower() == "none":
            self.dataset_class = StandardizedPlantCommunityDataset(
                None, None, None, is_dummy=True,
            )
            self._n_classes = 1
        elif StandardizedPlantCommunityDataset.is_dataset(self.args.dataset):
            self.dataset_class = StandardizedPlantCommunityDataset
            self._n_classes = len(StandardizedPlantCommunityDataset.get_class_names(self.args.dataset))
        else:
            raise ValueError("Invalid dataset name: " + self.args.dataset)

        if self.dataset_class.class_names is not None:
            self._n_classes = len(self.dataset_class.class_names)
        self.splits = self.dataset_class.get_splits(self.args.evaluation_mode)

        self.post_evaluation_task_types = ["cover_prediction"]
        self.post_evaluation_ts_task_types = ["cover_prediction_ts"]

    def add_info(self, **kwargs):
        super().add_info(**kwargs)

        # Invalidate previous instances of _configurations when new infos are added
        self._configurations = None

    @property
    def configurations(self):
        # Lazy loading
        if not self._configurations:
            self._configurations = self._init_subconfigs()

        return self._configurations

    def __getitem__(self, idx):
        return self._configurations[idx]

    @abstractmethod
    def _init_subconfigs(self):
        return []

    @property
    def n_model_outputs(self):
        return self._n_classes

    @property
    def post_evaluation_methods(self):
        if len(self.logger_tags) == 0:
            logger_tags = [["default"]]
        else:
            logger_tags = [[logger_tag] for logger_tag in self.logger_tags]

        evaluation_settings = self.dataset_class.EVALUATION_SETTINGS

        subtask_types = [None] + list(evaluation_settings.average_modes.keys())
        task_types = deepcopy(self.post_evaluation_task_types)

        n_subtask_types = len(subtask_types)
        n_task_types = len(task_types)

        # Create all combinations of subtasks and tasks
        subtask_types = subtask_types * n_task_types
        task_types = join([v] * n_subtask_types for v in task_types)

        if self.include_phenology:
            pheno_subtasks = []
            for pheno_type in ["flowering", "senescence"]:
                for subtask_type in subtask_types:
                    pheno_subtask = f"{pheno_type}_{subtask_type}" if subtask_type is not None else pheno_type
                    pheno_subtasks.append(pheno_subtask)
                    task_types.append("phenology")

            subtask_types.extend(pheno_subtasks)
            # task_types.extend(["phenology", "phenology"])
            # subtask_types.extend(["flowering", "senescence"])

        if self.eval_args.ts_eval:
            ts_subtask_types = deepcopy(evaluation_settings.ts_intermediate_average_keys)
            # Add prefix to intermediate subtasks
            ts_subtask_types = ["inter" + t if t is not None else None for t in ts_subtask_types]
            ts_task_types = self.post_evaluation_ts_task_types * len(ts_subtask_types)

            n_ts_subtask_types = len(ts_subtask_types)
            n_ts_task_types = len(ts_task_types)

            # Create all combinations
            ts_subtask_types = ts_subtask_types * n_ts_task_types
            ts_task_types = join([v] * n_ts_subtask_types for v in ts_task_types)

            if self.include_phenology:
                for inter_avg_key in evaluation_settings.ts_intermediate_average_keys:
                    prefix = f"{'inter' + inter_avg_key}_" if inter_avg_key is not None else ""
                    ts_task_types.extend(["phenology_ts", "phenology_ts"])
                    ts_subtask_types.extend([
                        prefix + "flowering",
                        prefix + "senescence",
                    ])

            subtask_types = subtask_types + ts_subtask_types
            task_types = task_types + ts_task_types

        print("Task types to evaluate:", task_types)
        print("Subtask types to evaluate:", subtask_types)

        eval_sets = None

        if hasattr(self.dataset_class, "get_eval_sets"):
            eval_sets = self.dataset_class.get_eval_sets(self.args.evaluation_mode)

        if not eval_sets:
            class_names = self.dataset_class.class_names_evaluation

            if class_names is None:
                class_names = self.dataset_class.get_class_names(self.args.dataset)

            eval_sets = {
                None: {
                    "plots": None,
                    "class_names": class_names,
                }
            }

        print("Eval sets:", eval_sets)

        evals = []

        for eval_set_name, eval_set_spec in eval_sets.items():
            eval_set_name_str = "_" + eval_set_name if eval_set_name else ""

            eval_plots = eval_set_spec["plots"]
            eval_class_sets = eval_set_spec["class_names"]

            eval_scale_values = self.dataset_class.EVAL_SCALE_VALUES["std"]
            eval_scale_kind = self.args.dataset + "_std"

            if isinstance(eval_class_sets, dict):
                class_set_suffixes = list(eval_class_sets.keys())
                class_names_list = list(eval_class_sets.values())
            else:
                class_set_suffixes = [""]
                class_names_list = [eval_class_sets]

            for task_type, subtask_type in zip(task_types, subtask_types):
                print(f"Evaluating task: {task_type}, subtask: {subtask_type}, eval set: {eval_set_name}")
                for logger_tag in logger_tags:
                    tag = logger_tag[0]
                    subtask_name = "_" + subtask_type if subtask_type else ""
                    data_folder = os.path.join(
                        self.output_folder,
                        "data",
                        self.experiment_name,
                        tag,
                    )
                    for class_names, class_name_suffix in zip(class_names_list, class_set_suffixes):
                        if task_type.startswith("cover_prediction"):
                            evals.extend([
                                BioEvaluation(
                                    name=f"bio_eval_{task_type}_{mode}_{tag + subtask_name}{eval_set_name_str}{class_name_suffix}",
                                    experiment_name=self.experiment_name,
                                    data_folder=data_folder,
                                    evaluation_mode=self.args.evaluation_mode,
                                    split_names=self.splits,
                                    include_dead_litter=False,
                                    run_dca=True,
                                    mode=mode,
                                    file_prefix="_" + task_type + "_" + tag + subtask_name,
                                    scale_kind=eval_scale_kind,
                                    scale_values=eval_scale_values,
                                    log_file="logs/cover_dca_procrustes_log.txt",
                                    logger_tags=[self.experiment_name] + logger_tag,
                                    class_names=class_names,
                                    plots=eval_plots,
                                ) for mode in ("mean", "concat")
                            ])

                        predictions_files = []
                        targets_files = []
                        indices_files = []

                        for split_name in self.splits:
                            file_pattern = os.path.join(
                                data_folder,
                                f"{split_name}_{task_type}_{tag}" + subtask_name
                            )

                            predictions_files.append(file_pattern + "_predictions.csv")
                            targets_files.append(file_pattern + "_targets.csv")
                            indices_files.append(file_pattern + "_indices.csv")

                        evals.append(
                            CorrelationEvaluation(
                                name=f"corr_eval_{task_type}_{tag + subtask_name}{eval_set_name_str}{class_name_suffix}",
                                predictions_file=predictions_files,
                                targets_file=targets_files,
                                indices_file=indices_files,
                                names=self.splits,
                                class_names=class_names,
                                plots=eval_plots,
                            )
                        )

            evals.append(
                MetricAggregation(
                    "cover_prediction_" + tag,
                    "cover_prediction_" + tag,
                    folder=self.metric_path,
                    subfolder_names=self.splits,
                )
            )

            if evaluation_settings.segmentation_evaluation:
                evals.append(
                    MetricAggregation(
                        "segmentation_" + tag,
                        "segmentation_" + tag,
                        folder=self.metric_path,
                        subfolder_names=self.splits,
                    )
                )

        return evals