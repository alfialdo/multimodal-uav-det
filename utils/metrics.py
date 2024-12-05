# TODO: add metrics to calculate FPS, GFLOPS ?
import torch
from torchvision.ops import complete_box_iou_loss, box_convert
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F

def bbox_loss(preds_decoded, targets, bbox_loss_fn='mse'):
    """
    Calculate the complete IoU loss between predicted and target bounding boxes.
    
    Args:
        preds_decoded (torch.Tensor): Predicted bounding boxes in format [n_anchors, h, w, 4] 
                                    where 4 represents box coordinates in cxcywh format
        targets (torch.Tensor): Target bounding boxes in format [N, 4] where N is number of ground truth boxes
                              and 4 represents box coordinates in cxcywh format
        head_size (int): Size of the feature map grid for this detection head
        head_anchors (torch.Tensor): Anchor boxes for this detection head in format [n_anchors, 2]
                                   where 2 represents width and height

    Returns:
        torch.Tensor: Complete IoU loss value between matched predictions and targets
    """ 
    if bbox_loss_fn == 'mse':
        bbox_loss = F.mse_loss(preds_decoded, targets, reduction='mean')

    elif bbox_loss_fn == 'ciou':
        # Convert format cxcywh to xyxy
        preds_decoded = box_convert(preds_decoded, in_fmt='cxcywh', out_fmt='xyxy')
        targets = box_convert(targets, in_fmt='cxcywh', out_fmt='xyxy')    


        # Calculate complete IoU loss between pred-target pairs, reduction use to get scalar value
        bbox_loss = complete_box_iou_loss(preds_decoded, targets, reduction='mean')

    return bbox_loss


def objectness_loss(preds_obj, targets, obj_scale_w, reduction='mean'):
    """
    Calculate binary cross entropy loss between predicted objectness scores and target values.
    
    Args:
        preds_obj (torch.Tensor): Predicted objectness scores in format [n_anchors, h, w, 1]
        targets (torch.Tensor): Target objectness values in format [n_anchors, h, w, 1] 
                              containing binary values (0 or 1)
        obj_scale_w (float): Weight factor to scale the objectness loss
        reduction (str, optional): Specifies the reduction to apply to the output.
                                 Can be 'none', 'mean', or 'sum'. Default: 'mean'
    
    Returns:
        torch.Tensor: Weighted binary cross entropy loss between predictions and targets.
                     If reduction is 'none', returns loss for each element.
                     If reduction is 'mean' or 'sum', returns reduced loss value.
    """
    preds_obj = preds_obj.squeeze(dim=-1)
    avg_loss = F.binary_cross_entropy_with_logits(preds_obj, targets, reduction=reduction)

    return avg_loss * obj_scale_w


def no_obj_loss(preds_no_obj, targets, reduction='mean'):
    """
    Calculate binary cross entropy loss between predicted objectness scores and target values
    for grid cells that don't contain objects.
    
    Args:
        preds_obj (torch.Tensor): Predicted objectness scores in format [n_anchors, h, w, 1]
                                 for grid cells without objects
        targets (torch.Tensor): Target objectness values in format [n_anchors, h, w, 1]
                              containing binary values (0 or 1) for grid cells without objects
        reduction (str, optional): Specifies the reduction to apply to the output.
                                 Can be 'none', 'mean', or 'sum'. Default: 'mean'
    
    Returns:
        torch.Tensor: Binary cross entropy loss between predictions and targets for no-object cells.
                     If reduction is 'none', returns loss for each element.
                     If reduction is 'mean' or 'sum', returns reduced loss value.
    """
    preds_no_obj = preds_no_obj.squeeze(dim=-1)
    avg_loss = F.binary_cross_entropy_with_logits(preds_no_obj, targets, reduction=reduction)

    return avg_loss



def calculate_ap(pred_boxes, pred_obj, target_boxes, max_det=300, iou_th=None):
    """
    Calculate Mean Average Precision (mAP) for object detection predictions.
    
    Args:
        preds (List[dict]): List of predicted bounding boxes in format [N, 4] where N is number of predictions
                                 and 4 represents box coordinates in cxcywh format
        target_boxes (torch.Tensor): Target bounding boxes in format [M, 4] where M is number of ground truth boxes
                                   and 4 represents box coordinates in cxcywh format
        max_det (int, optional): Maximum number of detections to consider. Default: 300
        iou_th (List[float], optional): List of IoU thresholds to evaluate AP. 
                                      If None, uses thresholds from 0.5 to 0.95 with step 0.05.
                                      Default: None
            
    Returns:
        dict: Dictionary containing AP metrics including:
            - map: Mean AP across IoU thresholds [0.5:0.95]
            - map_50: AP at IoU threshold 0.5
            - map_75: AP at IoU threshold 0.75
            - map_small: AP for small objects
            - map_medium: AP for medium objects  
            - map_large: AP for large objects
    """
    if iou_th is None:
        iou_th = [0.5 + 0.05 * i for i in range(10)]

    metric_ap = MeanAveragePrecision(
        box_format='cxcywh',
        iou_thresholds=iou_th,
        max_detection_thresholds=[max_det]*3
    )

    device = target_boxes.device

    pred_dict = [dict(
        boxes= pred_boxes,
        scores= pred_obj,
        labels= torch.ones(len(pred_boxes), dtype=torch.int64, device=device)
    )]
    
    target_dict = [dict(
        boxes= target_boxes,
        labels= torch.ones(len(target_boxes), dtype=torch.int64, device=device)
    )]
    
    # Update and compute metrics
    metric_ap.update(pred_dict, target_dict)

    return metric_ap.compute()