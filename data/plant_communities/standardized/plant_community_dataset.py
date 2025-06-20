import os
import json
import numpy as np
import pandas as pd

from PIL import Image

if __name__ == "__main__":
    import sys
    sys.path.append("../../..")


from data.plant_communities.vegetation_dataset import VegetationDataset


def get_sites(split, split_path):
    assert split in ("train", "validation")

    with open(os.path.join(split_path, "split_trainval.json"), "r") as f:
            split_dict = json.loads(f.read())

    sites = split_dict[split]

    return sites

class StandardizedPlantCommunityDataset(VegetationDataset):
    class_names = None
    class_names_evaluation = None
    class_has_phenology = None
    eval_sets = None
    index_names = ("site", "year", "month", "day", "hour", "minute")
    site_index = ("site",)
    time_index = ("year", "month", "day", "hour", "minute")

    EVAL_SCALE_VALUES = {
        "mean": 0,
        "std": 0,
    }

    @staticmethod
    def is_dataset(dataset_name):
        return os.path.exists(os.path.join(os.path.dirname(__file__), "../../../datasets/", dataset_name.split(",")[0]))

    @staticmethod
    def get_class_names(dataset_name):
        print("Dataset name is", dataset_name)
        with open(os.path.join(os.path.dirname(__file__), "../../../datasets/", dataset_name.split(",")[0], "class_names.txt"), "r") as f:
            class_names = f.read().strip().splitlines()

        return class_names

    def __init__(self, folder_name, split, mode, apply_masks=True, pheno_model="fs", is_dummy=False, **kwargs):
        super().__init__(data_kind=mode, **kwargs)

        if is_dummy:
            return # Dummy dataset, no data loading

        dataset_path = os.path.join(os.path.dirname(__file__), f"../../../datasets/{folder_name}")
        self.dataset_path = dataset_path

        self.class_names = self.get_class_names(folder_name)
        self.class_names_evaluation = self.class_names
        self.class_has_phenology = [1] * len(self.class_names)

        self.pheno_model = pheno_model
        self.apply_masks = apply_masks

        self.pheno_value_names = {
            0: "flowering",
            1: "senescence",
        }

        self.image_folder = os.path.join(dataset_path, "images")
        self.mask_folder = os.path.join(dataset_path, "masks")
        self.data_folder = os.path.join(dataset_path, "annotations")
        self.split_folder = os.path.join(dataset_path, "splits")

        self.sites = get_sites(split, self.split_folder)

        self.index_data = None
        self.cover_data = None
        self.phenology_data = None

        self.dataset_means = None
        self.dataset_stds = None

        self.tuple_order = "il"

        self.initialize_data()

    def initialize_data(self):
        with open(os.path.join(self.dataset_path, "data.json"), "r") as f:
            data = f.read()
            self.index_data = json.loads(data)[self.data_kind]

        self.cover_data = pd.read_csv(os.path.join(self.data_folder, "cover_" + self.data_kind + ".csv"), index_col=self.index_names) / 100
        self.cover_data = self.cover_data.fillna(0)

        if self.pheno_model not in (None, "none"):
            self.flowering_data = pd.read_csv(os.path.join(self.data_folder, "flowering_" + self.data_kind + ".csv"), index_col=self.index_names) / 100
            self.senescence_data = pd.read_csv(os.path.join(self.data_folder, "senescence_" + self.data_kind + ".csv"), index_col=self.index_names) / 100

            self.flowering_data = self.flowering_data.fillna(0)
            self.senescence_data = self.senescence_data.fillna(0)
        else:
            self.flowering_data = None
            self.senescence_data = None

        self.data_list = []
        self.source_indices = []
        self.original_data_points = []

        for site in self.sites:
            data_list = self.index_data[site]

            for data_point in data_list:
                image_path = os.path.join(self.image_folder, data_point["image"])
                mask_path = os.path.join(self.mask_folder, data_point["mask"])
                index = data_point["index"]

                self.original_data_points.append(True)

                data_index = (
                    index["site"],
                    index["year"],
                    index["month"],
                    index["day"],
                    index["hour"],
                    index["minute"],
                )

                target_data = []

                target_data.append(self.cover_data.loc[data_index].to_numpy())

                if self.pheno_model == "fs":
                    flow_data = self.flowering_data.loc[data_index].to_numpy()
                    sen_data = self.senescence_data.loc[data_index].to_numpy()

                    target_data.append(np.stack([flow_data, sen_data], axis=1))
                elif self.pheno_model not in (None, "none"):
                    raise ValueError("Invalid phenology model: ", self.pheno_model)

                self.data_list.append((
                    (image_path, mask_path),
                    tuple(target_data),
                ))
                self.source_indices.append(data_index)

        label_vals = []

        for inputs, targets in self.data_list:
            if isinstance(targets, (tuple, list)):
                targets = targets[0]
            #sum_vector += vals
            label_vals.append(targets)

        self.dataset_means = np.mean(label_vals, axis=0)
        self.dataset_stds = np.std(label_vals, axis=0)

        print(f"Dataset Means: {self.dataset_means}")
        print(f"Dataset STDs: {self.dataset_stds}")

    @staticmethod
    def get_splits(evaluation_mode):
        if evaluation_mode == "val":
            splits = ["val"]
        else:
            raise ValueError("Invalid eval mode: " + evaluation_mode)

        return splits

    def _preprocess_inputs(self, inputs, transform):
        image = None

        for input in inputs:
            image_path, mask_path = inputs

            image = self.image_server[image_path]

            if self.apply_masks:
                mask = self.image_server[mask_path].convert("L")
                image = Image.composite(image, Image.new("RGB", image.size), mask)

        if self.transform:
            image = self.transform(image)

        return image


#class DummyStandardizedPlantCommunityDataset(StandardizedPlantCommunityDataset):
