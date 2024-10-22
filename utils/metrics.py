# TODO: add metrics to calculate AP & AR (0.5 - 0.95), model params, FPS, GFLOPS
from torchvision.ops import box_convert, complete_box_iou_loss
import torch.nn.functional as F
import einops

def iou_loss(pred, target, reduction='mean'):
    # Convert x, y, w, h to x1, y1, x2, y2 using PyTorch's box conversion utility
    target = box_convert(target, in_fmt='xywh', out_fmt='xyxy')
    target = einops.rearrange(target, 'b xyxy -> b 1 xyxy')

    return complete_box_iou_loss(pred, target, reduction=reduction)

def cross_entropy_loss(pred, target):
    target = einops.rearrange(target, 'obj -> obj 1')
    return F.cross_entropy(pred, target)
