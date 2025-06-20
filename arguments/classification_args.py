from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


arguments = [
    Argument("learning_rate", type=float, default=0.001),
    Argument("weight_decay", type=float, default=1e-6),
    Argument("batch_size", type=int, default=12),
    Argument("nep", "n_epochs", type=int, default=None),
    Argument("ds", "dataset", type=str, default=""),
    Argument("mode", type=str, default="multiclass"),
    Argument("image_size", type=int, default=448, help="The image size for training."),
    Argument("inference_image_size", type=int, nargs="+", default=None, help="The image size for inference. If None, will default to 448 px for the shorter side, and a multiple of 32 for the longer side, approximately keeping aspect ratio."),
    Argument("output_images", type=str2bool, default=True, help="Whether to output images during inference."),
    Argument("use_occlusion_augmentation", type=str2bool, default=False, help="Whether to use occlusion augmentation (Cutout) during training."),
    Argument("threshold", type=float, nargs="+", default=0.2, help="The threshold for CAM binary mask generation. If multiple values are provided, segmentation will be done for each value."),
    Argument("normalization", type=str, choices=("torch", "caffe"), default="torch"),
    Argument("ewo", "enriched_wsol_output", type=str2bool, default=False, help="Whether to use enriched WSOL output (with test-time augmentation)."),
    Argument("loss", type=str, default="cce"),

    Argument("aug_strat", type=str, default="rot"),

    Argument("t", "tag", type=str, default=""),
]

classification_training_arg_handler = ArgumentHandler(
    prefix="c_",
    arguments=arguments,
)

model_arguments = [
       Argument("pooling_method", type=str, default="gap", help="Pooling method to use for the classification model. Options: 'gap' (Global Average Pooling), 'gmp' (Global Max Pooling), 'glsep' (Global Log-Sum-Exp Pooling))."),
]


classification_model_arg_handler = ArgumentHandler(
    prefix="cm_",
    arguments=model_arguments,
)