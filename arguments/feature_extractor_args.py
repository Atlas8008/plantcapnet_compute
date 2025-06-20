from .argument_handling import Argument, ArgumentHandler
from utils.utils import str2bool


arguments = [
    Argument("base_network", type=str, default="resnet50", help="Base network architecture; either a name in the torchvision model zoo or a path to a custom model."),
    Argument("use_fpn", type=str2bool, default="True", help="Use Feature Pyramid Network (FPN) for the base network."),
    Argument("fpn_spec", type=str, default="P2-512", help="FPN layer and feature specification, e.g., P2-512, P3-128, P4-64, P5-512."),
    Argument("fpn_unetlike", type=str2bool, default=False, help="Use UNet-like upsampling for the FPN architecture."),
    Argument("fpn_output_agg", type=str, default=None, help="If multiple FPN layer outputs are used, this specifies how to aggregate them. Options: 'none', 'add', 'mean'"),
    Argument("fas", "fpn_atrous_spec", type=str, default=None, help="Atrous FPN specification, will employ atrous FPN if specified."),
]


wsol_fe_arg_handler = ArgumentHandler(
    prefix="wf_",
    arguments=arguments
)
segmentation_fe_arg_handler = ArgumentHandler(
    prefix="sf_",
    arguments=arguments
)