from typing import NamedTuple
import torch

class DetectionResults(NamedTuple):
    bbox: torch.Tensor
    obj: torch.Tensor

class BatchData(NamedTuple):
    image: torch.Tensor
    bbox: torch.Tensor
    obj: torch.Tensor