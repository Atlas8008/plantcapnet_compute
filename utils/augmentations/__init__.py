from .utils import __as_transform

from .preprocessing import Preprocess, DePreprocess
from .noop import noop_transform
from .crop_and_pad import pad_to_size
from .cutout import cutout, inverted_cutout, generate_validation_occlusion_map, validation_cutout
from .nonlinear_color_transform import nonlinear_color_transform as nonlinear_color_transform_
from .imgaug import ImgaugWrapper
from .jigsaw import AdvancedRandomJigsaw
from .simple_augs import *

cutout_transform = __as_transform(cutout)
inverted_cutout_transform = __as_transform(inverted_cutout)
validation_cutout_transform = __as_transform(validation_cutout)
pad_to_size_transform = __as_transform(pad_to_size)
nonlinear_color_transform = __as_transform(nonlinear_color_transform_)
