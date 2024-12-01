from typing import NamedTuple, Union, List
import torch

class DetectionResults(NamedTuple):
    bbox: torch.Tensor
    obj: torch.Tensor

class BatchData(NamedTuple):
    image: torch.Tensor
    bbox: Union[torch.Tensor, List[torch.Tensor]]
    # obj: torch.Tensor

class Config:
    def __init__(self, cfg):
        self.set_attr(cfg)

    def set_attr(self, d, prefix=''):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                setattr(self, key, Config(v))
            else:
                setattr(self, key, v)