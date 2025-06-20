from .cover_prediction import CoverPredictionMetaConfiguration, CoverPredictionConfiguration
from .zeroshot import ZeroShotCoverPredictionMetaConfiguration
from .temporal_cover_prediction import TemporalCoverPredictionMetaConfiguration


def get_configurations(experiment_name, zeroshot_args, cover_args, temporal_cover_args, cover_eval_args):

    zeroshot_config = ZeroShotCoverPredictionMetaConfiguration(
        experiment_name,
        zeroshot_args,
        cover_eval_args,
    )
    cover_config = CoverPredictionMetaConfiguration(
        experiment_name,
        cover_args,
        cover_eval_args,
    )
    temporal_cover_config = TemporalCoverPredictionMetaConfiguration(
        cover_config,
        experiment_name,
        temporal_cover_args,
        cover_eval_args,
    )

    return {
        "zeroshot": zeroshot_config,
        "cover": cover_config,
        "temporal_cover": temporal_cover_config,
    }