import torch
import torchmetrics

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from copy import deepcopy
from multiprocessing import cpu_count

from utils import augmentations as augs

from losses import BCEWithScalarsAndLogitsLoss
from utils.torch_utils.torch_utils import set_trainability
from .configuration import Configuration

import cap_utils as lu

from data import GBIFDataset

from utils import torch_utils
from inference import wsol


def get_configurations(experiment_name, args):
    config = ClassificationConfiguration(experiment_name, args)

    return {
        "wsol": config,
    }


class ClassificationConfiguration(Configuration):
    def __init__(self, experiment_name, args):
        super().__init__(experiment_name, args)

        self.img_augmentations, _ = lu.get_augmentations(
            args.aug_strat,
            int(args.image_size / 0.875),
            force_resize=True,
        )

        self.args.dataset = self.args.dataset.lower()
        #assert self.args.dataset in GBIF_DEFAULT_NAMES_TO_PATHS, "Unknown dataset name: " + args.dataset
        self._n_classes = self.training_dataset.get_num_classes()

        #self._acc = torchmetrics.Accuracy()
        if self.args.mode == "multilabel":
            self._acc = torchmetrics.Accuracy(
                task="multilabel", num_labels=self._n_classes)
        elif self.args.mode == "multiclass":
            self._acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=self._n_classes)
        else:
            raise ValueError("Invalid mode: ", self.args.mode)

        self.task_name = "wsol"

    @property
    def n_model_outputs(self):
        return self._n_classes

    @property
    def training_transforms(self):
        return transforms.Compose([
            transforms.Resize(int(1.5 * self.args.image_size), interpolation=transforms.InterpolationMode.BILINEAR)
            ] +
            self.img_augmentations + [
            transforms.RandomCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
            augs.cutout_transform(min_mask_size=int(0.01 * 448), max_mask_size=int(0.35 * 448)) if self.args.use_occlusion_augmentation
                else augs.noop_transform(),
            transforms.ToTensor(),
            augs.Preprocess(self.args.normalization),
        ])

    @property
    def eval_transforms(self):
        return transforms.Compose([
            transforms.Resize(int(self.args.image_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            augs.Preprocess(self.args.normalization),
        ])

    @property
    def inference_transforms(self):
        pass

    @property
    def loss(self):
        if self.args.loss == "cce":
            return nn.CrossEntropyLoss()
        elif self.args.loss == "bce":
            return BCEWithScalarsAndLogitsLoss(self.training_dataset.get_num_classes())
        elif self.args.loss == "bce_bin":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Invalid loss: " + self.args.loss)

    @property
    @Configuration.cache
    def training_dataset(self):
        return GBIFDataset(
            "train",
            dataset_path=self.args.dataset,
            transform=self.training_transforms,
            cache_files=False
        )

    @property
    @Configuration.cache
    def eval_dataset(self):
        return GBIFDataset(
            "validation",
            dataset_path=self.args.dataset,
            transform=self.eval_transforms,
            cache_files=False
        )

    @property
    def training_loader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=min(cpu_count(), 16),
            drop_last=True,
            pin_memory=True
        )

    @property
    def eval_loader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=min(cpu_count(), 16),
            pin_memory=True
        )

    @property
    def model_kwargs(self):
        return {"head": "classification", "mode": "classification"}

    @property
    def metrics(self):
        return [self._acc]

    @property
    def scheduler_metric(self):
        return self._acc

    @property
    def early_stopping(self):
        if self.args.n_epochs is None or self.args.n_epochs < 0:
            return torch_utils.EarlyStopping(
                metric=self._acc,
                patience=6,
                mode="max",
            )

        return None

    @property
    def epochs(self):
        if self.args.n_epochs is None or self.args.n_epochs < 0:
            return 1000

        return self.args.n_epochs

    @property
    def inference_methods(self):
        from utils.wsol import segmentation_resize_augmentations
        from utils.augmentations import Preprocess

        resize_to_original = False

        segmentation_transforms = transforms.Compose(
            segmentation_resize_augmentations(self.args.image_size) + [
                transforms.ToTensor(),
                Preprocess(self.args.normalization),
        ])

        datasets = [
            GBIFDataset(
                "train",
                dataset_path=self.args.dataset,
                transform=segmentation_transforms,
            ),
            GBIFDataset(
                "validation",
                dataset_path=self.args.dataset,
                transform=segmentation_transforms,
            ),
        ]

        return [
            wsol.WSOLInference(
                thresholds=self.args.threshold,
                datasets=datasets,
                enriched_wsol_output=self.args.enriched_wsol_output,
                save_folder=self.output_path,
                resize_to_original=resize_to_original,
                model_kwargs={
                    "head": "classification",
                    "mode": "cam",
                }
            )
        ]

    def restore_model_from_checkpoint(self, model):
        m = torch.load(self.model_save_path, map_location="cpu", weights_only=False)

        model.feature_extractor.load_state_dict(
            m.feature_extractor.state_dict()
        )
        if model.classification_module is not None:
            model.classification_module.load_state_dict(
                m.classification_module.state_dict()
            )
        else:
            model.classification_module = deepcopy(m.classification_module)

    def set_model_trainability(self, model):
        set_trainability(model, True)

    def get_optimizer(self, model):
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

    def get_scheduler(self, optimizer):
        if self.args.n_epochs is None or self.args.n_epochs < 0:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=4,
            )
        else:
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                [int(self.args.n_epochs * 0.75),
                 int(self.args.n_epochs * 0.9)]
            )