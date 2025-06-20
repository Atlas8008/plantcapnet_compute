from .plant_cover import MeanScaledAbsoluteErrorLoss
from .segmentation import DiceLoss, ClassFocusedBCEDice
from .classification import BCEWithScalarsAndLogitsLoss, BCEFocalLoss

from .util import CombinedLoss, cast_method_inputs, squeeze_method_inputs, argmax_onehot, argmax
