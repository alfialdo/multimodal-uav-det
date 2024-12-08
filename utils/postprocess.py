import cv2
import numpy as np
import torch
import einops

from torchvision.ops import box_iou, box_convert

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, label=None, format='xyxy'):
    """Draw bounding box on image from xyxy format coordinates
    Args:
        image (np.ndarray): Image to draw on
        bbox (list or np.ndarray): Bounding box coordinates
        color (tuple): BGR color for bbox (default: green)
        thickness (int): Line thickness
        label (str, optional): Label text to draw above bbox
        format (str): Format of bbox coordinates - either 'xyxy' [x1,y1,x2,y2] or 'xywh' [x,y,w,h]
        
    Returns:
        np.ndarray: Image with drawn bbox
    """
    # Convert bbox format if needed
    if format == 'xywh':
        x, y, w, h = map(int, bbox)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
    else:  # xyxy format
        x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label is not None:
        # Get text size and set background size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1-text_height-baseline-5), (x1+text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x1, y1-baseline-3), font, font_scale, (255,255,255), 1)
        
    return image


def calculate_iou(preds, targets, head_anchors, mask=None):
    """Calculate IoU between predicted and target bounding boxes in grid cell format
    
    Args:
        preds (torch.Tensor): Predicted bounding boxes [N,4] in format (offset_x, offset_y, grid_cell_w, grid_cell_h)
        targets (torch.Tensor): Target bounding boxes [N,4] in format (offset_x, offset_y, grid_cell_w, grid_cell_h) 
        anchors (torch.Tensor): Anchor boxes [3,2] in format (w,h)
        mask (torch.Tensor): Mask for valid grid cells [N]
    Returns:
        torch.Tensor: IoU scores between predictions and targets [N]
    """
    # Scale preds width and height to grid cell size
    # Reshape anchors to match prediction shape for broadcasting
    device = preds.device
    anchors = einops.repeat(head_anchors, 'n_anchors wh -> n_anchors 1 1 wh').to(device)
    pred_bboxes = preds.detach().clone()
    # pred_bboxes[..., 2:] = pred_bboxes[..., 2:] * anchors

    # Convert bbox format from cxcywh to xyxy
    if mask is not None:
        pred_bboxes = pred_bboxes[mask]
        targets = targets[mask]
    else:
        pred_bboxes = einops.rearrange(pred_bboxes, 'n_anchors h w bbox -> (n_anchors h w) bbox')
        targets = einops.rearrange(targets, 'n_anchors h w bbox -> (n_anchors h w) bbox')

    pred_bboxes = box_convert(pred_bboxes, in_fmt='cxcywh', out_fmt='xyxy')
    targets = box_convert(targets, in_fmt='cxcywh', out_fmt='xyxy')

    # Calculate IoU
    ious = box_iou(pred_bboxes, targets)

    return ious[:,0]








