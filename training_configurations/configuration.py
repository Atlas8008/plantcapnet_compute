import os
import json
import hashlib

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


def _hash_args(args, additional_info):
    args = {k: args[k] for k in sorted(args.keys())}
    additional_info = {k: additional_info[k] for k in sorted(additional_info.keys())}

    res = hashlib.md5((str(args) + str(additional_info)).encode())

    return res.hexdigest()


class MetaConfiguration(ABC):
    def __init__(self, experiment_name, args, additional_info=None) -> None:
        super().__init__()

        self.experiment_name = experiment_name
        self.args = args
        self._additional_info = additional_info or {}
        self._output_folder = "outputs/"

        self.task_name = ""

        self._configurations = []

    def add_info(self, **kwargs):
        self._additional_info.update(kwargs)

    @property
    @abstractmethod
    def n_model_outputs(self):
        return None

    @property
    def configurations(self):
        return self._configurations

    @property
    def post_evaluation_methods(self):
        return []

    @property
    def post_inference_methods(self):
        return []

    @property
    def output_folder(self):
        return os.path.join(self._output_folder, self.task_name)

    @property
    def metric_folder(self):
        return os.path.join(self._output_folder, self.task_name, "metrics")

    @property
    def metric_path(self):
        return os.path.join(self.metric_folder, self.experiment_name)


class Configuration(ABC):
    def __init__(self, experiment_name, args):
        self.experiment_name = experiment_name
        self.args = args

        self._output_folder = "outputs/"
        self._model_folder = "saved_models/"

        self._cache = {}
        self._additional_info = {}
        self._meta_info = {}

        self.task_name = ""

        #self.arg_hash = _hash_args(args)

    @property
    def arg_hash(self) -> str:
        return _hash_args(
            {k: getattr(self.args, k) for k in sorted(vars(self.args))},
            {k: self._additional_info[k] for k in sorted(self._additional_info)},
        )

    @property
    def param_info(self) -> Dict[Any, Any]:
        hd = {"hash": self.arg_hash}

        return {**vars(self.args), **self._additional_info, **hd}

    def save_hash_and_config(self):
        hash_folder = "meta/hashes/" + self.task_name
        experiment_folder = "meta/experiments/" + self.task_name

        os.makedirs(hash_folder, exist_ok=True)
        os.makedirs(experiment_folder, exist_ok=True)

        param_info_dict = {
            "parameters": self.param_info,
            "additional_info": self._additional_info,
            "meta_info": self._meta_info,
        }

        with open(f"{hash_folder}/{self.arg_hash}.json", "w") as f:
            f.write(json.dumps(param_info_dict, indent=True))
        with open(f"{experiment_folder}/{self.experiment_name}.json", "w") as f:
            f.write(json.dumps(param_info_dict, indent=True))

    def clear_cache(self):
        self._cache = {}

    def add_info(self, **kwargs):
        self._additional_info.update(kwargs)

    def add_meta_info(self, **kwargs):
        self._meta_info.update(kwargs)

    @staticmethod
    def cache(fn) -> Callable[..., Any]:
        def cached_fn(self):
            cache_name = "__" + fn.__name__

            if not cache_name in self._cache:
                ret = fn(self)
                self._cache[cache_name] = ret

            return self._cache[cache_name]
        return cached_fn

    @property
    @abstractmethod
    def n_model_outputs(self) -> int:
        return 0

    @property
    @abstractmethod
    def training_transforms(self) -> Any:
        pass

    @property
    def training_target_transforms(self) -> Any:
        return None

    @property
    @abstractmethod
    def eval_transforms(self) -> Any:
        pass

    @property
    def eval_target_transforms(self) -> Any:
        return None

    @property
    def inference_transforms(self) -> Any:
        return []

    @property
    @abstractmethod
    def training_dataset(self) -> Any:
        pass

    @property
    @abstractmethod
    def eval_dataset(self) -> Any:
        pass

    @property
    def inference_dataset(self) -> Any:
        return None

    @property
    @abstractmethod
    def training_loader(self) -> Any:
        pass

    @property
    @abstractmethod
    def eval_loader(self) -> Any:
        pass

    @property
    def inference_loader(self):
        return None

    @property
    @abstractmethod
    def model_kwargs(self):
        pass

    @property
    @abstractmethod
    def loss(self):
        pass

    @property
    @abstractmethod
    def metrics(self):
        pass

    @property
    def scheduler_metric(self):
        return None

    @property
    def early_stopping(self):
        return None

    @property
    def epochs(self):
        return 0

    @property
    def evaluation_methods(self):
        return []

    @property
    def inference_methods(self):
        return []

    @property
    def meta_info_str(self):
        if self._meta_info:
            return "_".join(v for k, v in self._meta_info.items())

        return ""

    @property
    def model_folder(self):
        return os.path.join(self._model_folder, self.task_name)

    @property
    def model_save_path(self):
        return os.path.join(
            self.model_folder, self.arg_hash + self.meta_info_str + ".pth")

    @property
    def keep_saved_model(self):
        return True

    @property
    def model_link_path(self):
        return os.path.join(
            "meta/model_links",
            self.task_name,
            self.experiment_name + self.meta_info_str + ".pth"
        )

    @property
    def output_folder(self):
        return os.path.join(self._output_folder, self.task_name)

    @property
    def output_path(self):
        return os.path.join(self.output_folder, self.arg_hash + self.meta_info_str)

    @property
    def metric_folder(self):
        return os.path.join(self.output_folder, "metrics")

    @property
    def metric_path(self):
        return os.path.join(self.metric_folder, self.experiment_name)

    def restore_model_from_checkpoint(self, model):
        raise NotImplementedError()

    @abstractmethod
    def set_model_trainability(self, model):
        pass

    @abstractmethod
    def get_optimizer(self, model):
        pass

    @abstractmethod
    def get_scheduler(self, optimizer):
        pass

    def modify_model_with_configs(self, model, config_dict, device):
        return model

    def maybe_build_ensemble(self, model, config_dict, device):
        return model, False