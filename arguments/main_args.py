from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


arguments = [
    Argument("wsol_training", type=str2bool, default=True),
    Argument("wsol_evaluation", type=str2bool, default=None),
    Argument("wsol_inference", type=str2bool, default=None),
    Argument("wsol_restore_checkpoint", type=str, default=None),

    Argument("segmentation_training", type=str2bool, default=True),
    Argument("segmentation_evaluation", type=str2bool, default=None),
    Argument("segmentation_inference", type=str2bool, default=None),
    Argument("segmentation_restore_checkpoint", type=str, default=None),

    Argument("deocclusion_training", type=str2bool, default=False),
    Argument("deocclusion_evaluation", type=str2bool, default=None),
    Argument("deocclusion_inference", type=str2bool, default=None),
    Argument("deocclusion_restore_checkpoint", type=str, default=None),

    Argument("joint_deocclusion_training", type=str2bool, default=False),
    Argument("joint_deocclusion_evaluation", type=str2bool, default=None),
    Argument("joint_deocclusion_inference", type=str2bool, default=None),
    Argument("joint_deocclusion_restore_checkpoint", type=str, default=None),

    Argument("zeroshot_training", type=str2bool, default=True),
    Argument("zeroshot_evaluation", type=str2bool, default=None),
    Argument("zeroshot_inference", type=str2bool, default=None),
    Argument("zeroshot_restore_checkpoint", type=str, default=None),

    Argument("cover_training", type=str2bool, default=True),
    Argument("cover_evaluation", type=str2bool, default=True),
    Argument("cover_inference", type=str2bool, default=True),
    Argument("cover_restore_checkpoint", type=str, default=None),

    Argument("temporal_cover_training", type=str2bool, default=False),
    Argument("temporal_cover_evaluation", type=str2bool, default=False),
    Argument("temporal_cover_inference", type=str2bool, default=False),
    Argument("temporal_cover_restore_checkpoint", type=str, default="cover"),

    Argument("only_train_nonexisting", type=str2bool, default=False, help="Will check before training, if the resulting model already exists and will only train models that do not exist yet."),
    Argument("training_sparse_info", type=str2bool, default=False, help="If set, only the first and last batch of an epoch will be logged/printed. This can be used to reduce the amount of logging information to prevent logging clutter."),

    Argument("gpu", type=str, default="0"),
    Argument("multigpu", type=str2bool, default=False),

    Argument("tag", type=str, default=""),

    Argument("experiment_name", type=str, default="experiment"),
]

main_arg_handler = ArgumentHandler(
    prefix="m_",
    arguments=arguments
)