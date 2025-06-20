from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


seg_arguments = [
    Argument("learning_rate", type=float, default=0.001),
    Argument("weight_decay", type=float, default=1e-6),
    Argument("batch_size", type=int, default=12),
    Argument("nep", "n_epochs", type=int, default=None),
    Argument("ds", "dataset", type=str, default="__wsol__", help="The name of the dataset to use for training. If set to '__wsol__', the dataset will be the same as the one used during classification training and WSOL inference."),
    Argument("image_size", type=int, default=448, help="The size of the images to use for training."),
    Argument("image_size_p", type=int, default=None, help="Optional image size after data augmentation. If set, the images will be resized to this size after augmentation. If not set, the images will be resized to the original image size."),
    Argument("inference_image_size", type=int, nargs="+", default=None),
    Argument("normalization", type=str, choices=("torch", "caffe"), default="torch", help="Normalization method to use for the images. 'torch' is the default PyTorch normalization, 'caffe' is the Caffe normalization."),
    Argument("output_segmentations", type=str2bool, default=False),
    Argument("include_bg", type=str2bool, default=False, help="If set to True, the background class will be included in the segmentation output. If set to False, the background class will not be included in the segmentation output."),
    Argument("use_zoomin_aug", type=str2bool, default=False, help="Flag; if set, the ZoomIn augmentation will be applied to the images."),

    Argument("zoomin_min", type=float, default=0.5, help="The smallest image size after the ZoomIn augmentation."),

    Argument("uc", "use_cutout", type=str2bool, default=False, help="If set to True, the Inverted Cutout augmentation will be applied to the images. If set to False, the Inverted Cutout augmentation will not be applied."),
    Argument("ic_min", type=int, default=64, help="Minimum size of the Inverted Cutout cutout."),
    Argument("ic_max", type=int, default=448, help="Maximum size of the Inverted Cutout cutout."),
    Argument("ic_mask", type=str2bool, default=False),
    Argument("ic_reps", type=int, default=1, help="Number of repetitions of the Inverted Cutout cutout. If set to 1, only one cutout is applied. If set to 2, two cutouts are applied, etc."),
    Argument("ic_cut_segmap", type=str2bool, default=False),
    Argument("ic_indep", type=str2bool, default=False, help="If set to True, the Inverted Cutout cutout is applied independently for width and height. If set to False, the cutout is applied with the same size for both width and height."),
    Argument("icc", type=str2bool, default=False, help="If set, the IC cutout will be round instead of rectangular."), # IC circular
    Argument("icsvc", "ic_seg_visible_crop", type=str2bool, default=False),
    Argument("ic_config", type=str, default="icrdm", help="Configuration for the Inverted Cutout (IC) augmentation. Possible values: "
        "'ic' - Standard (Segmentation-based) IC augmentation, "
        "'icrdm' - (Segmentation-based) IC with random application, "
        "'vanic' - Vanilla IC without segmentation, "
        "'vanicrdm' - Vanilla IC with random application, "
        "'icscrdm' - IC also applied to the gt segmentation map with random application, "
        "'scic' - IC also applied to the gt segmentation."),

    Argument("monitor", type=str, choices=("iou", "dice"), default="iou", help="Metric to monitor during training. 'iou' is Intersection over Union, 'dice' is Dice coefficient."),

    Argument("fllf", "fpn_layers_lr_factor", type=float, default=1.0, help="Factor by which the learning rate for the FPN layers is multiplied. If set to 1.0, the learning rate for the FPN layers is the same as for the rest of the network. If set to 0.1, the learning rate for the FPN layers is 10 times smaller than for the rest of the network."),

    Argument("image_caching", type=str2bool, default=True, help="If set to True, the images will be cached in memory. This can speed up training, but requires more memory."),
    Argument("max_workers", type=int, default=16),

    Argument("loss", type=str, choices=("dice", "bce", "bce_dice", "focal", "cfbce_dice", "mse"),  default="bce_dice"),
    Argument("aug_strat", type=str, default="rot", help="Augmentation strategy to use during training. For the details, please check cap_utils/utils."),
    Argument("use_segmix", type=str2bool, default=False),
    Argument("ds_large_images", type=str2bool, default=False),

    Argument("eval_cutout", type=str2bool, default=False),

    Argument("t", "tag", type=str, default=""),
]

segmentation_training_arg_handler = ArgumentHandler(
    prefix="s_",
    arguments=seg_arguments
)
deoc_training_arg_handler = ArgumentHandler(
    prefix="d_",
    arguments=seg_arguments
)
joint_deoc_training_arg_handler = ArgumentHandler(
    prefix="j_",
    arguments=seg_arguments
)

segmentation_model_arguments = [
    Argument("segm_module_spec", type=str, default=None),
]

deoc_model_arguments = [
    Argument("deoc_module_spec", type=str, default=None),
    Argument("training_output_fuse_method", type=str, default="none"),
    Argument("eval_output_fuse_method", type=str, default="none"),
]

segmentation_model_arg_handler = ArgumentHandler(
    prefix="sm_",
    arguments=segmentation_model_arguments,
)
deoc_model_arg_handler = ArgumentHandler(
    prefix="dm_",
    arguments=deoc_model_arguments,
)