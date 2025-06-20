from torch import nn

import models

from .modules.cover_prediction import get_data_injection_module

def build_model(
        mode,
        n_outputs_pretraining,
        n_outputs_target_task,
        args,
    ):

    wsol_fe_model_args=vars(args["wsol_fe_model"])
    if args["segmentation_fe_model"].fpn_spec == "__wsol__":
        args["segmentation_fe_model"].fpn_spec = args["wsol_fe_model"].fpn_spec
        args["segmentation_fe_model"].use_fpn = args["wsol_fe_model"].use_fpn

    segmentation_fe_model_args=vars(args["segmentation_fe_model"])

    classification_model_args=vars(args["classification_model"])
    segm_model_args=vars(args["segm_model"])
    deoc_model_args=vars(args["deoc_model"])
    zeroshot_module_args=vars(args["zeroshot_module"])
    cover_model_args=vars(args["cover_model"])
    temporal_cover_model_args=vars(args["temporal_cover_model"])

    if mode == "wsol":
        # Training block for WSOL + Segmentation
        feature_extractor_module, n_feats_wsol = models.get_feature_extractor(
            **wsol_fe_model_args,
        )
        classification_module = models.get_classification_module(
            in_feats=n_feats_wsol,
            out_feats=n_outputs_pretraining,
            **classification_model_args,
        )

        model = models.PlantAnalysisModel(
            feature_extractor=feature_extractor_module,
            classification_module=classification_module,
        )
    else:
        feature_extractor_module, n_feats_segm = models.get_feature_extractor(
            **segmentation_fe_model_args,
        )

        segmentation_module = models.get_segmentation_module(
            in_feats=n_feats_segm,
            out_feats=n_outputs_pretraining,
            segmentation_module_args=segm_model_args,
            deoc_module_args=deoc_model_args,
        )
        zeroshot_cover_prediction_module = models.get_segmentation_based_cover_prediction_module(
            **zeroshot_module_args,
        )
        cover_prediction_module = models.get_cover_prediction_module(
            in_feats=n_feats_segm,
            out_feats=n_outputs_target_task,
            **cover_model_args,
        )
        if temporal_cover_model_args["m_spec"] is not None:
            temporal_cover_fusion_module = get_data_injection_module(
                n_outputs=n_feats_segm,
                spec=temporal_cover_model_args["m_spec"],
            )
        else:
            temporal_cover_fusion_module = None

        model = models.PlantAnalysisModel(
            feature_extractor=feature_extractor_module,
            segmentation_module=segmentation_module,
            zeroshot_cover_module=zeroshot_cover_prediction_module,
            cover_module=cover_prediction_module,
            temporal_cover_fusion_module=temporal_cover_fusion_module,
        )

        zeroshot_cover_prediction_module.model = (model, )
        if hasattr(cover_prediction_module, "set_model"):
            cover_prediction_module.set_model(model)

    return model


class DistributedParallelWrapper(nn.parallel.DistributedDataParallel):
    # Expose functionality of underlying model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)