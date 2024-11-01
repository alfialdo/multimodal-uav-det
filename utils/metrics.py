# TODO: add metrics to calculate FPS, GFLOPS ?
import torch
from torchvision.ops import box_convert, complete_box_iou_loss, box_iou
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
import einops


def bbox_loss(preds, targets):
    """
    Calculate the average bounding box loss between predicted and target boxes.
    
    Args:
        preds (torch.Tensor): Predicted bounding boxes in format [N, 4] where N is number of predictions
                            and 4 represents box coordinates
        targets (torch.Tensor): Target bounding boxes in format [M, 4] where M is number of ground truth boxes
                              and 4 represents box coordinates

    Returns:
        torch.Tensor: Average loss value across all valid predictions and targets. Returns 0.0 if no valid
                     predictions exist.

    Note:
        For each target box, finds the minimum complete IoU loss with all predicted boxes.
        The final loss is the mean of these minimum losses across all targets.
    """
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
    """
    Calculate binary cross entropy loss between predicted objectness scores and target values.
    
    Args:
        preds (torch.Tensor): Predicted objectness scores in range [0,1]
        targets (torch.Tensor): Target objectness values (0 or 1)
        reduction (str, optional): Specifies the reduction to apply to the output.
                                 Can be 'none', 'mean', or 'sum'. Default: 'mean'
    
    Returns:
        torch.Tensor: Binary cross entropy loss between predictions and targets.
                     If reduction is 'none', returns loss for each element.
                     If reduction is 'mean' or 'sum', returns reduced loss value.
    """
    # Ensure target is on same device as pred
    avg_loss = F.binary_cross_entropy(preds, targets, reduction=reduction)

    return avg_loss



def calculate_ap(pred_boxes, pred_objs, target_boxes, max_det=150, iou_th=None):
    """
    Calculate Mean Average Precision (mAP) for RTMUAVDet predictions.
    
    Args:
        predictions (List[DetectionResults]): List of detection results from different scales
            Each DetectionResults contains:
            - bbox: tensor of shape [B, A, H, W, 4] in (cx, cy, w, h) format
            - obj: tensor of shape [B, A, H, W, 1] containing objectness scores
            
        targets (BatchData): Batch of ground truth data containing:
            - bbox: List of tensors, each of shape [M, 4] in (x1, y1, x2, y2) format
            
    Returns:
        dict: Dictionary containing various AP metrics including 'map' (AP@[0.5:0.95])
    """
    if iou_th is None:
        iou_th = [0.5 + 0.05 * i for i in range(10)]

    metric_ap = MeanAveragePrecision(
        box_format='xyxy',
        iou_thresholds=iou_th,
        max_detection_thresholds=[max_det]*3,
    )

    pred_dict = [dict( 
        boxes= pred_boxes,
        scores= pred_objs,
        labels= torch.ones(len(pred_boxes), dtype=torch.int64, device=pred_boxes.device)
    )]
    
    target_dict = [dict(
        boxes= target_boxes,
        labels= torch.ones(len(target_boxes), dtype=torch.int64, device=target_boxes.device)
    )]
    
    # Update and compute metrics
    metric_ap.update(pred_dict, target_dict)

    return metric_ap.compute()

def filter_high_iou_bboxes(pred_bboxes, pred_objs, target_bboxes, iou_th=0.5):
    """
    Filter predicted bounding boxes based on IoU with target boxes.
    
    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes in xyxy format, shape [N, 4]
        pred_objs (torch.Tensor): Predicted objectness scores, shape [N]
        target_bboxes (torch.Tensor): Target bounding boxes in xyxy format, shape [M, 4]
        iou_th (float, optional): IoU threshold for filtering. Defaults to 0.5.
        
    Returns:
        tuple:
            - filtered_boxes (torch.Tensor): Filtered predicted boxes with IoU > threshold
            - filtered_objs (torch.Tensor): Filtered objectness scores for remaining boxes
            - true_objectness (torch.Tensor): Binary mask same shape as pred_objs, 1.0 for 
              predictions with IoU > threshold, 0.0 otherwise
    """
    device = pred_bboxes.device
    ious = box_iou(pred_bboxes, target_bboxes) 
    
    # Get maximum IoU for each prediction
    max_ious, _ = torch.max(ious, dim=1)  # (1D)

    # Filter IoU than less than threshold (0.5)
    filter_iou = max_ious > iou_th
    filtered_boxes = pred_bboxes[filter_iou, :]
    filtered_objs = pred_objs[filter_iou]
    
    # Create objectness matrix (0 or 1) with original shape
    true_objectness = torch.zeros_like(pred_objs).to(device)
    true_objectness[filter_iou] = 1.0
    
    return filtered_boxes, filtered_objs, true_objectness