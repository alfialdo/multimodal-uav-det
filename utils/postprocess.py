import cv2
import numpy as np


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








