import abc

from data.default_dataset import DefaultDataset

from .vegetation_time_series import VegetationTimeSeriesDataset
from .evaluation_settings import EvaluationSettings


__all__ = ["VegetationDataset", "PlantCommunityDataset"]


class VegetationDataset(DefaultDataset, abc.ABC):
    """
    VegetationDataset is an abstract base class that represents a dataset of vegetation data.
    It inherits from DefaultDataset and abc.ABC.

    Attributes:
        class_names (list): List of class names.
        class_names_evaluation (list): List of class names used for evaluation.
        class_has_phenology (bool): Indicates if the class has phenology data.
        index_names (list): List of index names.
        site_index (int): Index of the site.
        time_index (int): Index of the time.
        EVAL_SCALE_VALUES (list): List of evaluation scale values.
        EVALUATION_SETTINGS (EvaluationSettings): Settings for evaluation.
    """
    class_names = None
    class_names_evaluation = None
    class_has_phenology = None
    index_names = None
    site_index = None
    time_index = None
    EVAL_SCALE_VALUES = None
    EVALUATION_SETTINGS = EvaluationSettings()

    def __init__(self, *args, data_kind=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_list = None
        self.source_indices = None
        self.original_data_points = None
        self.dataset_means = None
        self.dataset_stds = None
        self._data_kind = data_kind
        # self._eval_data_kind = eval_data_kind

    @abc.abstractmethod
    def initialize_data(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_splits(evaluation_mode):
        return None

    @property
    def data_kind(self):
        return self._data_kind

    @data_kind.setter
    def data_kind(self, val):
        self._data_kind = val

        self.initialize_data()

    def as_time_series(self, data_kind=None, *args, **kwargs):
        return VegetationTimeSeriesDataset(
            self,
            data_kind=data_kind,
            *args,
            **kwargs,
        )

    def __getitem__(self, idx):
        inputs, label = self._get_path_label_tuple(idx)

        inputs = self._preprocess_inputs(inputs, self.transform)
        label = self._preprocess_label(label, self.target_transform)

        return inputs, label

    def _preprocess_inputs(self, inputs, transform):
        image = self.image_server[inputs]

        if transform:
            image = transform(image)

        return image

    def _preprocess_label(self, label, transform):
        if transform:
            label = transform(label)

        return label

    @property
    def input_target_index_items(self):
        input_list = []
        target_list = []
        index_list = []

        for i in range(len(self.data_list)):
            input, label = self._get_path_label_tuple(i)
            index = self.source_indices[i]

            input_list.append(input)
            target_list.append(label)
            index_list.append(index)

        return input_list, target_list, index_list

    def get_num_classes(self):
        return len(self.class_names)

PlantCommunityDataset = VegetationDataset