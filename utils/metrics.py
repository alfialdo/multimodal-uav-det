# TODO: add metrics to calculate AP & AR (0.5 - 0.95), FPS, GFLOPS
import torch
from torchvision.ops import box_convert, complete_box_iou_loss
import torch.nn.functional as F
import einops

def iou_loss(pred, target, reduction='mean'):
    target = einops.rearrange(target, 'b xyxy -> b 1 xyxy')

    return complete_box_iou_loss(pred, target, reduction=reduction)

def cross_entropy_loss(pred, target):
    target = einops.rearrange(target, 'obj -> obj 1')
    return F.cross_entropy(pred, target)


from torchmetrics.detection import MeanAveragePrecision

def calculate_ap_ar_single_class(preds, targets):
    """
    Calculate Average Precision (AP) and Average Recall (AR) for object detection with a single class.
    
    :param preds: List of dictionaries, each containing 'boxes' and 'scores'
    :param targets: List of dictionaries, each containing 'boxes'
    :return: Dictionary containing AP and AR metrics
    """
    # Add dummy labels for single class (0)
    for pred in preds:
        pred['labels'] = torch.zeros_like(pred['scores'], dtype=torch.int64)
    for target in targets:
        target['labels'] = torch.zeros_like(target['boxes'][:, 0], dtype=torch.int64)

    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        class_metrics=True  # This will give us per-class metrics, which is what we want for single class
    )
    
    metric.update(preds, targets)
    results = metric.compute()
    
    return {
        'map': results['map'].item(),
        'map_50': results['map_50'].item(),
        'map_75': results['map_75'].item(),
        'mar_1': results['mar_1'].item(),
        'mar_10': results['mar_10'].item(),
        'mar_100': results['mar_100'].item(),
    }



