from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


arguments = [
    Argument("learning_rate", type=float, default=0.001),
    Argument("lr_scheduler", type=str, default="default"),
    Argument("weight_decay", type=float, default=1e-6),

    Argument("t", "tag", type=str, default="", help="Tag for the model. If not provided, the model will be saved with a default name."),

    Argument("nep", "n_epochs", type=int, default=0),

    Argument("evaluation_mode", type=str, choices=("val", "cv"), default="val", help="Evaluation mode. 'val' for validation, 'cv' for cross-validation."),

    Argument("dataset", type=str, default="", help="(Short) name or path to the dataset. If path, it should be a folder containing the dataset."),
    Argument("dataset_mode", type=str, choices=("weekly", "daily"), default="weekly"),
    Argument("pheno_model", type=str, choices=(None, "none", "fs"), default=None, help="The phenology data to be provided by the dataset. Currenly only 'fs' (flowering/senescence) is supported."),

    Argument("image_size", type=int, nargs="+", default=1536, help="The size of the input images (height, width). If a single value is provided, it will be used as the width, and the height will be set to half the width. If two values are provided, they will be used as the height and width respectively."),
    Argument("use_deocclusion_module", type=str2bool, default=False, help="Optional usage of a deocclusion module. Currently unused."),
    Argument("loss", type=str, default="mae"),
    Argument("pheno_loss", type=str, default="mae"),

    Argument("normalization", type=str, choices=("torch", "caffe"), default="torch", help="Normalization method for the input images. 'torch' uses the standard PyTorch normalization, while 'caffe' uses the Caffe normalization method."),
    Argument("enriched_eval", type=str2bool, default=False, help="If True, the model is evaluated using test time augmentation (TTA) with the provided enrichment parameters. The model is evaluated on the original images and the enriched images, and the results are combined."),
    Argument("inference_configurations", type=str, nargs="+", default="bzero,dnone", help="The configurations for the inference. The first part of the string (before the comma) is the specification of the background model, designated by 'b' (e.g., 'bzero' for no background model, i.e., background is predicted as zero). The second part of the string (after the comma) is the specification of the discretization of the class predictions (sigmoidal probabilities), designated with a 'd'. Can be 'none' (no discretization), 'argmax' (argmax discretization), 'scaled' (values are scaled to the range of 0-1 using the max value) or 'scaled_thresh' (values are scaled to the range of 0-1 using the max value and then thresholded at 0.5)."),

    Argument("enrichment_params", type=str, default="s1,s0.5,v,h", help="Parameters for test time augmentation. Can be a comma-separated list of values.\n\
                - 's<float>' - Scale factor. Can be a float or a comma-separated list of floats. E.g. 's1,s0.5'.\n\
                - 'v' - Vertical flip.\n\
                - 'h' - Horizontal flip."),
    Argument("interlace_size", type=int, default=512, help="Size of the patches to use for interlacing the images. Recommended to be 224, 256, 448 or 512"),

    Argument("batch_size", type=int, default=1),

    Argument("keep_saved_model", type=str2bool, default=False, help="If True, the model will not be deleted after training."),

    Argument("ensemble_tags", type=str, nargs="+", default=None),
    Argument("ensemble_meta_tag_suffixes", type=str, nargs="+", default=None),
    Argument("ensemble_model_set", type=str, default="none"),

    Argument("ensemble_model_names", nargs="*", type=str, default=None),

    Argument("max_workers", type=int, default=16),
]

zeroshot_arg_handler = ArgumentHandler(
    prefix="z_",
    arguments=arguments,
)

model_args = [
    Argument("flow_pheno_model", type=str, default=None, help="Path to or name of the flowering phenology model. If None, the model will not be used."),
    Argument("sen_pheno_model", type=str, default=None, help="Path to or name of the senescence phenology model. If None, the model will not be used."),
    Argument("relevance_model", type=str, default="none", help="Path to a relevance model differentiating the image into relevant and irrelevant areas. If 'none'', the entire image is considered relevant or masking has to be used to designate relevant regions."),
    Argument("relevance_model_preprocessing", type=str, default="caffe", help="Preprocessing method for the relevance model. 'caffe' uses the Caffe preprocessing method, while 'torch' uses the PyTorch preprocessing method."),
    Argument("preprocessing", type=str, default="caffe", help="Preprocessing method for the input images. 'caffe' uses the Caffe preprocessing method, while 'torch' uses the PyTorch preprocessing method."),
    Argument("class_mapper", type=str, default=None, help="Class mapper to use. Can be None, 'none' or the name of a predefined mapper. If None, the model will not be used. If 'none', the model will be used without a mapper. If a name is provided, the corresponding mapper will be used. The mappers are defined in the MAPPERS dictionary in 'models/modules/cover_prediction/asymm_segmentation_mappers.py.' Can map a list of classes to a subset of classes. This makes it possible to extract classifications for a subset of classes without changing the model architecture."),
]

zeroshot_module_arg_handler = ArgumentHandler(
    prefix="zm_",
    arguments=model_args
)