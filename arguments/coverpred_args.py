from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool, str2boolorstrfun


cover_training_args = [
    Argument("learning_rate", type=float, nargs="+", default=0.001, help="Learning rate(s) for the optimizer."),
    Argument("lr_scheduler", type=str, default="default", help="Learning rate scheduler to use."),
    Argument("weight_decay", type=float, default=1e-6, help="Weight decay (L2 penalty) for optimizer."),
    Argument("nep", "n_epochs", type=int, default=30, help="Number of epochs to train."),

    Argument("evaluation_mode", type=str, default="val", choices=("val", "cv"), help="Evaluation mode. 'val' for validation, 'cv' for cross-validation."),

    Argument("dataset", type=str, default="ia", help="(Short) name or path to the dataset. If path, it should be a folder containing the dataset."),
    Argument("dataset_mode", type=str, choices=("weekly", "daily"), default="weekly", help="Dataset mode: 'weekly' or 'daily'."),
    Argument("pheno_model", type=str, choices=(None, "none", "vfs", "fs"), default=None, help="The phenology data to be provided by the dataset. Currenly only 'fs' (flowering/senescence) is supported."),

    Argument("image_size", type=int, nargs="+", default=1536, help="The size of the input images (height, width). If a single value is provided, it will be used as the width, and the height will be set to half the width. If two values are provided, they will be used as the height and width respectively."),
    Argument("loss", type=str, default="mae", help="Loss function to use."),
    Argument("pheno_loss", type=str, default="mae", help="Loss function for phenology prediction."),

    Argument("output_folder", type=str, default="./outputs/cover"),
    Argument("normalization", type=str, choices=("torch", "caffe"), default="torch", help="Normalization method for the input images. 'torch' uses the standard PyTorch normalization, while 'caffe' uses the Caffe normalization method."),

    Argument("train_base_model", type=str2bool, default=False),

    Argument("mc_params", type=str, default=None),

    Argument("aug_strat", type=str, default="hflip"),

    Argument("use_combined_model", type=str2bool, default=False),
    Argument("ensemble_tags", type=str, nargs="+", default=None),
    Argument("ensemble_epochs", type=int, nargs="+", default=None),
    Argument("ensemble_meta_tag_suffixes", type=str, nargs="+", default=None),

    Argument("ensemble_model_names", nargs="*", type=str, default=None),

    Argument("batch_size", type=int, default=1, help="Batch size for training and evaluation."),

    Argument("t", "tag", type=str, default="", help="Tag for the model. If not provided, the model will be saved with a default name."),
    Argument("keep_saved_model", type=str2bool, default=False, help="If set, the model will not be deleted after training."),
    Argument("max_workers", type=int, default=16, help="Maximum number of worker processes for data loading."),
]

cover_model_args = [
    Argument("cover_head_type", type=str, default="default", help="Type of head to use for the cover prediction model. Can be 'default' (sigmoidal model with irrelevance prediction), 'simple_sigmoid' (sigmoidal model without irrelevance), 'simple_softmax' (softmax model without background class), 'simple_softmax2' (softmax model with background class) or 'linear' (linear model with global average pooling applied)."),
    Argument("pheno_model", type=str, default=None),
    Argument("dinj_spec", type=str, default=None, help="Specification of the data injection module that can be used to, for example, inject sensor information into the network. This is a string that is passed to the data injection module. Refer to the documentation of the data injection module for more information."),
    Argument("pheno_prediction_mode", type=str, default=None, help="Prediction mode for the phenology model. Can be 'classwise', 'joint', 'jointad', 'jointad2', 'naive' or any of 'jointad3', 'joint_imp', 'combined', 'joint++', 'joint_improved'."),
]

cover_eval_args = [
    Argument("t", "tag", type=str, default="", help="Tag for the model. If not provided, the model will be saved with a default name."),

    Argument("enriched_eval", type=str2bool, default=False, help="If True, the model is evaluated using test time augmentation (TTA) with the provided enrichment parameters. The model is evaluated on the original images and the enriched images, and the results are combined."),
    Argument("ts_eval", type=str2bool, default=False, help="If True, a time series model is used an evaluated on extracted time series data."),
    Argument("ts_train_data_kind", type=str, default="auto", help="Kind of data to use for training the time series model. Can be 'auto' (default), 'weekly', or 'daily'."),
    Argument("ts_eval_data_kind", type=str, default="weekly", help="Kind of data to use for evaluating the time series model. Can be 'weekly' or 'daily'."),
    Argument("ts_model_spec", type=str, default=None, help="Specification of the time series model to use. Refer to the individual time series models (under models/time_series) for more information on the specification format."),
    Argument("ts_include_feats", type=str2bool, default=False, help="If True, the original network features are included in the time series model instead of only the final predictions."),

    Argument("sen_eval_mask", type=str, default=None),

    Argument("segmentation_eval", type=str2boolorstrfun("auto"), default="auto"),
    Argument("segmentation_inference", type=str2boolorstrfun("auto"), default="auto"),
    Argument("save_eval_segmaps", type=str2bool, default=True),
    Argument("save_eval_errmaps", type=str2bool, default=True),
]


cover_training_arg_handler = ArgumentHandler(
    prefix="cp_",
    arguments=cover_training_args,
)
cover_model_arg_handler = ArgumentHandler(
    prefix="cpm_",
    arguments=cover_model_args,
)
cover_eval_arg_handler = ArgumentHandler(
    prefix="ce_",
    arguments=cover_eval_args,
)