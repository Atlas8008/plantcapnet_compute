import argparse

from .argument_handling import Argument, ArgumentHandler
from .main_args import main_arg_handler
from .classification_args import classification_training_arg_handler, classification_model_arg_handler
from .segmentation_args import segmentation_training_arg_handler, deoc_training_arg_handler, joint_deoc_training_arg_handler, deoc_model_arg_handler, segmentation_model_arg_handler
from .coverpred_args import cover_training_arg_handler, cover_model_arg_handler, cover_eval_arg_handler
from .temporal_coverpred_args import temporal_cover_training_arg_handler, temporal_cover_model_arg_handler
from .feature_extractor_args import wsol_fe_arg_handler, segmentation_fe_arg_handler
from .zeroshot_args import zeroshot_arg_handler, zeroshot_module_arg_handler


def get_arguments():
    parser = argparse.ArgumentParser()

    main_arg_handler(parser)

    classification_training_arg_handler(parser)
    segmentation_training_arg_handler(parser)
    deoc_training_arg_handler(parser)
    joint_deoc_training_arg_handler(parser)
    zeroshot_arg_handler(parser)
    cover_training_arg_handler(parser)
    temporal_cover_training_arg_handler(parser)

    cover_eval_arg_handler(parser)

    wsol_fe_arg_handler(parser)
    segmentation_fe_arg_handler(parser)
    classification_model_arg_handler(parser)
    segmentation_model_arg_handler(parser)
    deoc_model_arg_handler(parser)
    zeroshot_module_arg_handler(parser)
    cover_model_arg_handler(parser)
    temporal_cover_model_arg_handler(parser)

    args = parser.parse_args()

    return {
        "main": main_arg_handler.get_args(args),
        "classification": classification_training_arg_handler.get_args(args),
        "segmentation": segmentation_training_arg_handler.get_args(args),
        "deocclusion": deoc_training_arg_handler.get_args(args),
        "joint_deocclusion": joint_deoc_training_arg_handler.get_args(args),
        "zeroshot_cover": zeroshot_arg_handler.get_args(args),
        "cover_training": cover_training_arg_handler.get_args(args),
        "temporal_cover_training": temporal_cover_training_arg_handler.get_args(args),

        "cover_eval": cover_eval_arg_handler.get_args(args),

        "wsol_fe_model": wsol_fe_arg_handler.get_args(args),
        "segmentation_fe_model": segmentation_fe_arg_handler.get_args(args),
        "classification_model": classification_model_arg_handler.get_args(args),
        "segm_model": segmentation_model_arg_handler.get_args(args),
        "deoc_model": deoc_model_arg_handler.get_args(args),
        "zeroshot_module": zeroshot_module_arg_handler.get_args(args),
        "cover_model": cover_model_arg_handler.get_args(args),
        "temporal_cover_model": temporal_cover_model_arg_handler.get_args(args),
    }