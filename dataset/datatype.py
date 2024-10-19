from typing import NamedTuple
import torch

class DetectionResults(NamedTuple):
    bbox: torch.Tensor
    obj: torch.Tensor