import re
import torch

from torch import nn
from torch.nn import functional as F


FINAL_IMPROVED_JOINT = (
    "jointad3", "joint_imp", "combined", "joint++", "joint_improved"
)


class PhenologyModule(nn.Module):
    """Class for phenology prediction, specifically for flowering and senescence stages.
    """
    def __init__(self, in_feats, n_classes, pheno_model="fs", pheno_prediction_mode="classwise"):
        """
        Args:
            in_feats (int): Number of input features.
            n_classes (int): Number of classes.
            pheno_model (str): Phenology model to use. Can be "fs" (flowering and senescence) or "vfs" (vegetative, flowering, and senescence).
            pheno_prediction_mode (str): Prediction mode for the phenology model. Can be "classwise", "joint", "jointad", "jointad2", "naive" or any of "jointad3", "joint_imp", "combined", "joint++", "joint_improved" for the improved joint model.
        """
        super().__init__()

        self.classifier_activation = torch.sigmoid
        self.detach = False
        self.naive = False

        in_feats_before = in_feats

        added_convs = []

        if pheno_model == "default":
            pheno_model = "fs_none_sigm_ac1"

        if "_" in pheno_model:
            specs = pheno_model.split("_")

            pheno_model = specs[0]
            sub_model = specs[1]

            if len(specs) > 2:
                for spec in specs[2:]:
                #act_fun = specs[2]

                    if spec == "hs":
                        self.classifier_activation = F.hardsigmoid
                    elif spec == "sigm":
                        self.classifier_activation = torch.sigmoid
                    elif spec == "det":
                        self.detach = True
                    elif match := re.match(r"ac([1-9])", spec): # Added convs
                        for _ in range(int(match.group(1))):
                            added_convs.append(
                                nn.Conv2d(
                                    in_feats_before, in_feats, kernel_size=(1, 1)
                                ),
                            )
                            added_convs.append(nn.ReLU(inplace=True))
                            in_feats_before = in_feats
                    else:
                        raise ValueError("Invalid spec: " + spec)
        else:
            sub_model = None

        self.n_classes = n_classes
        self.pheno_model = pheno_model
        self.sub_model = sub_model
        self.prediction_mode = pheno_prediction_mode
        self.n_stages = None
        self.n_predictors = None
        self.n_adaptors = None

        adaptor_convs = 0

        if self. prediction_mode == "naive":
            self.naive = True
            self.n_predictors = self.n_classes
        elif self.prediction_mode == "classwise":
            self.n_predictors = self.n_classes
        elif self.prediction_mode == "joint":
            self.n_predictors = 1
        elif self.prediction_mode == "jointad":
            self.n_predictors = 1
            self.n_adaptors = self.n_classes
            adaptor_convs = 1
        elif self.prediction_mode == "jointad2":
            self.n_predictors = 1
            self.n_adaptors = self.n_classes
            adaptor_convs = 2
        elif self.prediction_mode in FINAL_IMPROVED_JOINT: # Best performing one, as shown in Körschens, Matthias, et al. "Unified Automatic Plant Cover and Phenology Prediction." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
            self.n_predictors = 1
            self.n_adaptors = self.n_classes
            adaptor_convs = 2
        else:
            raise ValueError("Invalid prediction mode: " + self.prediction_mode)

        if pheno_model == "vfs": # Vegetative, flowering, senescence
            self.n_stages = 3
        elif pheno_model == "fs": # Flowering, senescence. vegetative is implicit; this is the default
            self.n_stages = 2
        else:
            raise ValueError("Invalid phenology model: " + pheno_model)

        self.convs = nn.Sequential(
            *added_convs,
            nn.Conv2d(in_feats_before, in_feats, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )
        in_feats_before = in_feats

        self.phenology_classifier = nn.Conv2d(
            in_feats,
            self.n_predictors * self.n_stages,
            kernel_size=(1, 1),
        )
        if self.n_adaptors is not None:
            layers = []

            for _ in range(adaptor_convs - 1):
                layers.append(
                    nn.Conv2d(
                        in_feats,
                        in_feats,
                        kernel_size=(1, 1),
                    )
                )
                layers.append(nn.ReLU(inplace=True))

            self.phenology_adaptor = nn.Sequential(
                *layers,
                nn.Conv2d(
                    in_feats,
                    self.n_adaptors * self.n_stages,
                    kernel_size=(1, 1),
                ),
            )
        else:
            self.phenology_adaptor = None

    def forward(self, classes, feats, img=None, mode="phenology_prediction", eps=1e-4):
        if self.detach:
            classes = classes.detach()
            feats = feats.detach()

        feats = self.convs(feats)
        pheno = self.phenology_classifier(feats)
        pheno = torch.reshape(
            pheno,
            (
                -1,
                self.n_predictors,
                self.n_stages,
                pheno.shape[2],
                pheno.shape[3],
            )
        )

        if self.phenology_adaptor is not None:
            adaptor_weights = self.phenology_adaptor(feats)
            adaptor_weights = torch.reshape(
                adaptor_weights,
                (
                    -1,
                    self.n_adaptors,
                    self.n_stages,
                    adaptor_weights.shape[2],
                    adaptor_weights.shape[3],
                )
            )
        else:
            adaptor_weights = None

        classes = torch.reshape(
            classes,
            (-1, self.n_classes, 1, classes.shape[2], classes.shape[3]),
        )

        if self.pheno_model == "vfs":
            veg_flow = self.classifier_activation(pheno[:, :, 0:2], dim=2)
            senescence = self.classifier_activation(pheno[:, :, 2:3])

            pheno = torch.concat([veg_flow, senescence], dim=2)
        elif self.pheno_model == "fs":
            if self.prediction_mode in ("jointad", "jointad2"):
                pheno = self.classifier_activation(pheno)
                adaptor_weights = torch.sigmoid(adaptor_weights)

                pheno = pheno * adaptor_weights
            elif self.prediction_mode in FINAL_IMPROVED_JOINT:
                pheno = pheno + adaptor_weights
                pheno = self.classifier_activation(pheno)
            else:
                pheno = self.classifier_activation(pheno)
        else:
            raise ValueError("Invalid phenology model: " + self.pheno_model)

        if not self.naive:
            corrected_classwise_pheno = classes * pheno

            pheno = corrected_classwise_pheno

        if isinstance(mode, tuple):
            was_multimode = True
        else:
            was_multimode = False
            mode = (mode,)

        outputs = {}

        for m in mode:
            if m == "phenology_prediction":
                pheno_vals = torch.sum(pheno, dim=(3, 4)) / (torch.sum(classes, dim=(3, 4)) + eps) # Add eps to prevent division by 0

                outputs[m] = pheno_vals
            elif m == "phenology_max":
                pheno_vals = torch.amax(pheno, dim=(3, 4))

                outputs[m] = pheno_vals
            elif m == "segmentation":
                outputs[m] = pheno
            else:
                raise ValueError("Invalid mode: " + mode)

        if was_multimode:
            return outputs
        else:
            assert len(outputs) == 1
            return list(outputs.values())[0]

