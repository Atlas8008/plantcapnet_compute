import torch
import torchmetrics

from torchvision import transforms
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from cap_utils import get_augmentations

from .configuration import Configuration
from utils import torch_utils
from utils import augmentations as augs
from utils.augmentations import torchaug

from data import GBIFDataset
from data.gbif import ImageZoomIn
from metrics.segmentation import ContinuousDiceScore
from losses import cast_method_inputs
from cap_utils import get_augmentations, get_loss


def get_configurations(
        experiment_name,
        segmentation_args,
        deocclusion_args,
        joint_args,
        segmentation_path,
    ):
    segmentation_config = SegmentationConfiguration(
        experiment_name,
        segmentation_args,
        mode="segmentation",
        segmentation_path=segmentation_path,
    )
    deocclusion_config = SegmentationConfiguration(
        experiment_name,
        deocclusion_args,
        mode="deocclusion",
        segmentation_path=segmentation_path,
    )
    joint_deocclusion_config = SegmentationConfiguration(
        experiment_name,
        joint_args,
        mode="joint",
        segmentation_path=segmentation_path,
    )

    return {
        "segmentation": segmentation_config,
        "deocclusion": deocclusion_config,
        "joint_deocclusion": joint_deocclusion_config,
    }


class SegmentationConfiguration(Configuration):
    def __init__(self, experiment_name, args, mode, segmentation_path):
        assert mode in ("segmentation", "deocclusion", "joint")
        super().__init__(experiment_name, args)

        force_resize = False

        print("force_resize is", force_resize)

        self.mode = mode
        self.img_augs, self.segm_augs = get_augmentations(args.aug_strat, args.image_size, force_resize=force_resize)

        self.args.dataset = self.args.dataset.lower()
        self.segmentation_path = segmentation_path

        iou = torchmetrics.JaccardIndex(task="multilabel", num_labels=self.training_dataset.get_num_classes())
        iou = cast_method_inputs(iou, "update", target_type=torch.int32)

        dice_score = ContinuousDiceScore()

        iou._expensive = True

        self._metrics = [iou, dice_score]

        if args.monitor == "iou":
            self._scheduler_metric = iou
        elif args.monitor == "dice":
            self._scheduler_metric = dice_score
        elif args.monitor == "loss":
            self._scheduler_metric = self.loss
        else:
            raise ValueError()

        self._n_classes = self.training_dataset.get_num_classes()

        self.task_name = mode

    @property
    def n_model_outputs(self):
        return self._n_classes

    @property
    def training_transforms(self):
        if self.args.image_size_p is not None:
            post_resize = [
                transforms.Resize(
                    self.args.image_size_p,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            ]
        else:
            post_resize = []

        ic_transforms = []

        if self.args.use_cutout:
            segmix_image_transforms = transforms.Compose(self.img_augs + [
                transforms.RandomCrop(self.args.image_size),
                transforms.RandomHorizontalFlip(),
            ])
            segmix_segmentation_transforms = transforms.Compose(
                [transforms.ToTensor()] +
                self.segm_augs + [
                transforms.RandomCrop(self.args.image_size),
                transforms.RandomHorizontalFlip(),
            ])
            if self.args.ic_config.startswith("vanic"):
                ic = augs.inverted_cutout_transform(
                    min_mask_size=self.args.ic_min,
                    max_mask_size=self.args.ic_max,
                    reps=self.args.ic_reps,
                    circular=self.args.icc,
                    independent_wh=self.args.ic_indep,
                )
            else:
                ic = augs.torchaug.SegmentationBasedInvertedCutoutFunctional(
                    min_mask_size=self.args.ic_min,
                    max_mask_size=self.args.ic_max,
                    reps=self.args.ic_reps,
                    circular=self.args.icc,
                    independent_wh=self.args.ic_indep,
                    seg_visible_crop=self.args.ic_seg_visible_crop,
                    segmix=self.args.use_segmix,
                    segmix_img_transform=segmix_image_transforms,
                    segmix_segmap_transform=segmix_segmentation_transforms,
                )

            if self.args.ic_config == "ic":
                ic_transforms = [{"with segmentation": ic}]
            elif self.args.ic_config == "icrdm":
                ic_transforms = [
                    {"with segmentation": torchaug.RandomChoiceImg([
                        {"with segmentation": ic},
                        augs.noop_transform(),
                    ])
                    }
                ]
            elif self.args.ic_config == "vanic": # Vanilla IC
                ic_transforms = [ic]
            elif self.args.ic_config == "vanicrdm": # Vanilla IC
                ic_transforms = [
                    transforms.RandomChoice([
                        ic,
                        augs.noop_transform(),
                    ])
                ]
            elif self.args.ic_config == "icscrdm": # Segmentation IC + Random Default Image
                ic_transforms = [torchaug.RandomChoiceImg([
                    {"with segmentation": ic},
                    {"with segmentation": torchaug.SegmentationCutout()},
                    augs.noop_transform(),
                ])]
            elif self.args.ic_config == "scic":
                ic_transforms = [
                    {"with segmentation": ic},
                    {"with segmentation": torchaug.SegmentationCutout()},
                ]
            else:
                raise ValueError("Invalid value for ic_config: " + self.args.ic_config)

        t = torchaug.SegmentationCompose(self.img_augs + [
                transforms.RandomCrop(self.args.image_size),
                transforms.RandomHorizontalFlip()
        ] + ic_transforms + post_resize + [
                transforms.ToTensor(),
                augs.Preprocess(self.args.normalization),
        ], collect_additional_outputs=True)

        print(t)

        return t

    @property
    def training_target_transforms(self):
        if self.args.image_size_p is not None:
            post_resize = [
                transforms.Resize(
                    self.args.image_size_p,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ]
        else:
            post_resize = []

        return transforms.Compose(
            [transforms.ToTensor()] +
            self.segm_augs + [
            transforms.RandomCrop(self.args.image_size),
            transforms.RandomHorizontalFlip(),
        ] + post_resize)

    @property
    def eval_transforms(self):
        if self.args.image_size_p is not None:
            post_resize = [
                transforms.Resize(
                    self.args.image_size_p,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            ]
        else:
            post_resize = []

        return transforms.Compose([
                transforms.CenterCrop(self.args.image_size),
                transforms.RandomChoice([
                    augs.torchaug.SegmentationBasedInvertedCutout(
                        min_mask_size=self.args.ic_min,
                        max_mask_size=self.args.ic_max,
                        reps=self.args.ic_reps,
                        circular=self.args.icc,
                    ),
                    augs.noop_transform(),
                ]) if self.args.eval_cutout else augs.noop_transform()
            ] +
            post_resize +
            [
                transforms.ToTensor(),
                augs.Preprocess(self.args.normalization),
        ])

    @property
    def eval_target_transforms(self):
        if self.args.image_size_p is not None:
            post_resize = [
                transforms.Resize(
                    self.args.image_size_p,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ]
        else:
            post_resize = []

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.args.image_size),
        ] + post_resize)

    @property
    def loss(self):
        loss = get_loss(self.args.loss)

        return loss

    @property
    def training_dataset(self):
        if self.args.use_zoomin_aug:
            zoomin_aug = ImageZoomIn(
                sampling_remove_index=0 if self.args.include_bg else None,
                scale=(self.args.zoomin_min, 1),
            )
        else:
            zoomin_aug = None

        return GBIFDataset(
            "train",
            dataset_path=self.args.dataset,
            segmentation_path=self.segmentation_path,
            transform=self.training_transforms,
            target_transform=self.training_target_transforms,
            cache_files=self.args.image_caching,
            include_background_class=self.args.include_bg,
            zoomin_augmentation=zoomin_aug,
            large_images=self.args.ds_large_images,
            ic_mask=self.args.ic_mask,
            ic_cut_segmap=self.args.ic_cut_segmap,
        )

    @property
    def eval_dataset(self):
        return GBIFDataset(
            "validation",
            dataset_path=self.args.dataset,
            segmentation_path=self.segmentation_path,
            transform=self.eval_transforms,
            target_transform=self.eval_target_transforms,
            cache_files=self.args.image_caching,
            include_background_class=self.args.include_bg,
        )

    @property
    def training_loader(self):
        return DataLoader(self.training_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=min(cpu_count(), self.args.max_workers), drop_last=True, pin_memory=True)

    @property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=min(cpu_count(), self.args.max_workers), pin_memory=True)

    @property
    def inference_loader(self):
        pass

    @property
    def model_kwargs(self):
        if self.mode == "segmentation":
            return {
                "head": "segmentation",
                "use_deocclusion": False,
                "segmentation_kwargs": {
                    "mode": "segmentation",
                }
            }
        else:
            return {
                "head": "segmentation",
                "use_deocclusion": True,
                "segmentation_kwargs": {
                    "mode": "segmentation",
                },
                "deocclusion_kwargs": {
                    "mode": "segmentation",
                }
            }

    @property
    def metrics(self):
        return self._metrics

    @property
    def early_stopping(self):
        if self.args.n_epochs is None or self.args.n_epochs == -1:
            return torch_utils.EarlyStopping(
                metric=self._scheduler_metric,
                patience=6,
                mode="max"
            )

        return None

    @property
    def epochs(self):
        if self.args.n_epochs is None or self.args.n_epochs == -1:
            return 1000
        elif self.args.n_epochs < -1000: # Special setting: if n_epochs is lower than -1000, the learning rate will be kept constant for the entire training duration and the model will be trained for -n_epochs - 1000 epochs (e.g. -2000 means 1000 epochs)
            # This is useful for experimental purposes, when we want to train the model for a long time without changing the learning rate
            return -self.args.n_epochs - 1000
        else:
            return self.args.n_epochs

    @property
    def evaluation_methods(self):
        return []

    def restore_model_from_checkpoint(self, model):
        print("Restoring model from", self.model_save_path)
        m = torch.load(self.model_save_path, map_location="cpu", weights_only=False)

        model.feature_extractor.load_state_dict(
            m.feature_extractor.state_dict()
        )
        if self.mode == "segmentation": # Load just the segmentation module part without the deocclusion part
            model.segmentation_module.segmentation_module.load_state_dict(
                m.segmentation_module.segmentation_module.state_dict()
            )
        elif self.mode in ("deocclusion", "joint"):
            model.segmentation_module.load_state_dict(
                m.segmentation_module.state_dict()
            )

        print("Restored model hash:", hash(str(list(model.parameters()))))

    def set_model_trainability(self, model):
        if self.mode == "segmentation":
            torch_utils.set_trainability(
                model,
                True,
                verbose=True,
            )
        elif self.mode == "deocclusion":
            torch_utils.set_trainability(
                model.feature_extractor,
                False,
            )
            torch_utils.set_trainability(
                model.segmentation_module.segmentation_module,
                False,
            )
            torch_utils.set_trainability(
                model.segmentation_module.deocclusion_module,
                True,
            )
            torch_utils.print_param_trainability(model)
        elif self.mode == "joint":
            torch_utils.set_trainability(
                model,
                True,
            )
            torch_utils.print_param_trainability(model)


    def get_optimizer(self, model):
        return torch.optim.AdamW(
            params=[
                {
                    "params": model.feature_extractor.base_model.parameters(),
                },
                {
                    "params": model.feature_extractor.fpn_layers.parameters(),
                    "lr": self.args.learning_rate * self.args.fpn_layers_lr_factor,
                },
                {
                    "params": model.segmentation_module.parameters(),
                    "lr": self.args.learning_rate * self.args.fpn_layers_lr_factor,
                }
            ],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

    @property
    def scheduler_metric(self):
        return self._scheduler_metric

    def get_scheduler(self, optimizer):
        if self.args.n_epochs is None  or self.args.n_epochs == -1:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=4,
            )

        # Epochs without scheduler
        if self.args.n_epochs < -1000:
            return None

        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [
                int(self.args.n_epochs * 2/3),
                int(self.args.n_epochs * 8/9)
            ])

    @property
    def inference_methods(self):
        return []