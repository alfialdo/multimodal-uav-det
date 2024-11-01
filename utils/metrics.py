# TODO: add metrics to calculate FPS, GFLOPS
import torch
from torchvision.ops import box_convert, complete_box_iou_loss
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
import einops


def bbox_loss(preds, targets):

    losses = []

    for t_bbox in targets:
        if len(preds) != 0:
            loss = complete_box_iou_loss(preds, t_bbox).min()
            losses.append(loss)
        else:
            continue
    
    if len(losses) == 0:
        avg_loss = torch.tensor(0.0)
    else:
        avg_loss = torch.tensor(losses).mean()

    return avg_loss

def objectness_loss(preds, targets, reduction='mean'):
    # Ensure target is on same device as pred
    preds = preds.squeeze()
    avg_loss = F.binary_cross_entropy(preds, targets, reduction=reduction)

    return avg_loss


def calculate_ap_ar_single_class(preds, targets):
    """
    Calculate Average Precision (AP) and Average Recall (AR) for object detection with a single class.
    
    :param preds: List of dictionaries, each containing 'boxes' and 'scores'
    :param targets: List of dictionaries, each containing 'boxes'
    :return: Dictionary containing AP and AR metrics
    """
    # Add dummy labels for single class (0)r
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
