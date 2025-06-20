from dataclasses import dataclass, field
from typing import List, Dict, Union

def nonelist_factory():
    return [None]

@dataclass
class EvaluationSettings:
    ts_intermediate_average_keys: List = field(default_factory=nonelist_factory)
    average_modes: Dict = field(default_factory=dict)
    segmentation_evaluation: bool = False
    segmentation_inference: bool = False
    segmentation_inference_images: Union[List, None] = None