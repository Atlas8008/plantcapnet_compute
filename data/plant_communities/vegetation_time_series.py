import torch
import pandas as pd

from copy import deepcopy
from collections import defaultdict
from torch.utils.data import Dataset

from data.utils import get_input_label_tuple

def _unsqueeze_all(t):
    if isinstance(t, tuple):
        t = tuple(tensor[None] for tensor in t)
    else:
        t = t[None]

    return t


class VegetationTimeSeriesDataset(Dataset):
    def __init__(self, veg_dataset, data_kind=None, transform=None, target_transform=None):
        super().__init__()

        self._dataset = deepcopy(veg_dataset)

        # Only change data_kind, if a specific value is specified, otherwise leave it as it is in the original dataset
        if data_kind not in (None, "auto"):
            self._dataset.data_kind = data_kind

        self.image_server = self._dataset.image_server
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = self._dataset.transform
        if self.target_transform is None:
            self.target_transform = self._dataset.target_transform

        self.data_list, self.source_indices, self.data_is_ground_truth = self.get_time_series(
            self._dataset.data_list,
            self._dataset.source_indices,
            self._dataset.original_data_points,
            as_list=True,
        )

        print(f"Constructed time series list with {len(self.data_list)} series")

    def _preprocess_inputs(self, t):
        inputs, label = get_input_label_tuple(
            t,
            self._dataset.tuple_order
        )
        inputs = self._dataset._preprocess_inputs(inputs, self.transform)

        #inputs = _unsqueeze_all(inputs)

        return inputs

    def _preprocess_labels(self, t):
        inputs, label = get_input_label_tuple(
            t,
            self._dataset.tuple_order
        )
        label = self._dataset._preprocess_label(
            label, self.target_transform)

        #label = _unsqueeze_all(torch.tensor(label))
        if isinstance(label, (tuple, list)):
            return tuple(torch.tensor(l) for l in label)
        else:
            return torch.tensor(label)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self._preprocess_inputs(self.data_list[idx[0]][idx[1]]), \
                self._preprocess_labels(self.data_list[idx[0]][idx[1]])
        else:
            def inputs_generator():
                for t in self.data_list[idx]:
                    yield _unsqueeze_all(self._preprocess_inputs(t))

            def labels_generator():
                for t in self.data_list[idx]:
                    yield _unsqueeze_all(self._preprocess_labels(t))

            return inputs_generator(), labels_generator()

    def get_time_series(self, data_list, source_indices, original_data_points, as_list=False):
        """
        Extracts and organizes time series data from the provided data list and source indices.

        Parameters:
        -----------
        data_list : list
            A list containing the data points to be organized into time series.
        source_indices : list of lists
            A list of lists where each sublist contains indices corresponding to the data points in `data_list`.
        original_data_points : list
            A list containing the original data points corresponding to the indices in `source_indices`.
        as_list : bool, optional
            If True, the resulting time series dictionaries will be converted to lists. Default is False.

        Returns:
        --------
        data_time_series : dict or list
            A dictionary (or list if `as_list` is True) where keys are site tuples and values are lists of data points from `data_list`.
        index_time_series : dict or list
            A dictionary (or list if `as_list` is True) where keys are site tuples and values are lists of index tuples.
        orig_data_points_time_series : dict or list
            A dictionary (or list if `as_list` is True) where keys are site tuples and values are lists of original data points.
        """
        # Extract the indices from the data_list
        indices = list(range(len(data_list)))

        single_indices_lists = {
            k: [source_indices[i][idx_idx] for i in indices] for idx_idx, k in enumerate(self._dataset.index_names)
        }

        single_indices_lists["index"] = indices

        site_index = self._dataset.site_index
        time_index = self._dataset.time_index

        if not isinstance(site_index, tuple):
            site_index = (site_index,)
        if not isinstance(time_index, tuple):
            time_index = (time_index,)

        index_names_full = tuple(self._dataset.index_names)

        df_indices = pd.DataFrame(
            {k: single_indices_lists[k] for k in (index_names_full + ("index",))}
        )

        index_names_sort = site_index + time_index

        # Sort the DataFrame by unit, cam, seq_id and week_id
        df_indices.sort_values(list(index_names_sort), inplace=True)

        # Create a dictionary for time series of original tuple values
        index_time_series = defaultdict(list)

        # Create a dictionary for time series of tuples from the second list
        data_time_series = defaultdict(list)
        orig_data_points_time_series = defaultdict(list)

        for _, row in df_indices.iterrows():
            site = tuple(row[list(site_index)].values.tolist())
            time = tuple(row[list(time_index)].values.tolist())
            index_values = tuple(row[list(index_names_full)].values.tolist())
            index = row["index"]

            # Add the original tuple values to the time series
            index_time_series[site].append(index_values)
            data_time_series[site].append(data_list[index])
            orig_data_points_time_series[site].append(original_data_points[index])

        if as_list:
            # Convert dictionaries to lists
            index_time_series = list(index_time_series.values())
            data_time_series = list(data_time_series.values())
            orig_data_points_time_series = list(orig_data_points_time_series.values())

        return data_time_series, index_time_series, orig_data_points_time_series

    def __len__(self):
        return len(self.data_list)

    def get_lengths(self):
        return [len(l) for l in self.data_list]

    @property
    def input_target_index_items(self):
        input_list = []
        target_list = []
        index_list = []

        for i, ts_list in enumerate(self.data_list):
            for j in range(len(ts_list)):
                input, label = get_input_label_tuple(
                    self.data_list[i][j],
                    self._dataset.tuple_order
                )
                index = self.source_indices[i][j]

                input_list.append(input)
                target_list.append(label)
                index_list.append(index)

        return input_list, target_list, index_list

    # Expose functionality of underlying dataset
    def __getattr__(self, name):
        return getattr(self._dataset, name)


class NPrevStepDatasetWrapper(Dataset):
    """
    A dataset wrapper that provides supplementary data from previous steps.

    Args:
        dataset (Dataset): The underlying dataset to wrap.
        n (int, optional): The number of previous steps to include as supplementary data. Default is 1.

    Methods:
        __getitem__(index):
            Retrieves the item at the given index along with supplementary data from previous steps.
            Args:
                index (tuple): A tuple containing the indices to retrieve data from.
            Returns:
                tuple: A tuple containing the data and the target, where the data includes the original data and the supplementary data if available.

        __getattr__(name):
            Exposes the functionality of the underlying dataset.
            Args:
                name (str): The attribute name to access from the underlying dataset.
            Returns:
                Any: The attribute value from the underlying dataset.
    """
    def __init__(self, dataset, n=1) -> None:
        super().__init__()

        self.dataset = dataset
        self.n = n

    def __getitem__(self, index):
        assert isinstance(index, tuple)

        x, y = self.dataset[index]

        # Supplementary data
        supp_indices = list(range(index[1] - self.n, index[1]))

        supp_x = []

        for idx in supp_indices:
            _x, _ = self.dataset[(index[0], idx)]

            supp_x.append(_x)

        if len(supp_x) > 0:
            if len(supp_x) == 1:
                supp_x = supp_x[0]
            else:
                supp_x = torch.stack(supp_x, dim=-1)

            x = {"x": x, "suppl_data": supp_x}

        return x, y

    # Expose functionality of underlying dataset
    def __getattr__(self, name):
        return getattr(self.dataset, name)
