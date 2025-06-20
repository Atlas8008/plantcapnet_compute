from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


temporal_cover_training_args = [
    Argument("learning_rate", type=float, nargs="+", default=1e-5),
    Argument("weight_decay", type=float, default=0.0),
    Argument("nep", "n_epochs", type=int, default=20),
    Argument("batch_size", type=int, default=1),
    Argument("n_time_steps", type=int, default=1),

    Argument("image_size", type=int, nargs="+", default=[1536]),
    Argument("loss", type=str, default="mae"),
    Argument("pheno_loss", type=str, default="mae"),

    Argument("output_folder", type=str, default="./outputs/temporal_cover"),
    Argument("normalization", type=str, choices=("torch", "caffe"), default="torch"),

    Argument("train_time_steps", type=str2bool, default=False),
    Argument("train_base_model", type=str2bool, default=False),
    Argument("tmp_share_weights", type=str2bool, default=True),

    Argument("mc_params", type=str, default=None),
    Argument("aug_strat", type=str, default="hflip"),
    Argument("use_combined_model", type=str2bool, default=False),

    Argument("keep_saved_model", type=str2bool, default=False),
    Argument("t", "tag", type=str, default=""),
]

temporal_cover_model_args = [
    Argument("m_spec", type=str, default=None),
]


temporal_cover_training_arg_handler = ArgumentHandler(
    prefix="tcp_",
    arguments=temporal_cover_training_args,
)
temporal_cover_model_arg_handler = ArgumentHandler(
    prefix="tcpm_",
    arguments=temporal_cover_model_args,
)