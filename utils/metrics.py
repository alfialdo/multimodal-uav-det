# TODO: add metrics to calculate FPS, GFLOPS
import torch
from torchvision.ops import box_convert, complete_box_iou_loss
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
import einops


def bbox_loss(preds, targets):

    # preds = box_convert(preds, in_fmt='cxcywh', out_fmt='xyxy')

    losses = []

    for t_bbox in targets.squeeze():
        # TODO: ciou loss calculation method
        if preds.size(1) != 0:
            loss = complete_box_iou_loss(preds.squeeze(), t_bbox).min()
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
    preds = einops.rearrange(preds, 'b total_pred obj -> (b total_pred obj)')
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


# def bbox_ciou(box1, box2, eps=1e-7):
#     """
#     Calculate CIoU loss between two bounding boxes
#     Args:
#         box1: Predictions (N, 4) in (cx, cy, w, h) format
#         box2: Ground truth (N, 4) in (cx, cy, w, h) format
#     """
#     # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
#     b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
#     b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
#     b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
#     b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

#     # Intersection area
#     inter = torch.clamp((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)), min=0) * \
#             torch.clamp((torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)), min=0)

#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
#     union = (w1 * h1 + eps) + w2 * h2 - inter

#     iou = inter / union

#     # Calculate the smallest enclosing box
#     cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # Convex width
#     ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # Convex height

#     # Calculate diagonal distance
#     c2 = cw ** 2 + ch ** 2 + eps
#     rho2 = ((box2[..., 0] - box1[..., 0]) ** 2 +
#             (box2[..., 1] - box1[..., 1]) ** 2)  # Center dist squared

#     # Calculate aspect ratio consistency term
#     v = (4 / (math.pi ** 2)) * \
#         torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    
#     # Calculate alpha term for aspect ratio consistency
#     alpha = v / (1 - iou + v + eps)

#     # Calculate CIoU
#     ciou = iou - (rho2 / c2 + v * alpha)
    
#     return ciou

