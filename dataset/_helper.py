import os
import io
import json
import paramiko
import numpy as np
import cv2
import joblib
from typing import Tuple
from dotenv import load_dotenv

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils.datatype import BatchData


def load_json(path, remote_client=None):
    """
    Load JSON data from a file.

    Args:
    path (str): The path to the JSON file.
    remote_client (object, optional): A client for remote file access. If None, local file system is used.

    Returns:
    dict: The loaded JSON data.

    This function can load JSON data from both local and remote files,
    depending on whether a remote_client is provided.
    """
    if remote_client:
        with remote_client.open(path, 'rb') as f:
            f.prefetch()
            data = f.read()
            data = json.load(io.BytesIO(data))
    else:
        with open(path, 'r') as f:
            data = json.load(f)

    return data


def load_attributes(path, remote_client=None):
    """
    Load attribute files from a given path.

    This function reads JSON files containing attribute information for different labels.
    Each file in the specified path is expected to be a JSON file with a label name as its filename.

    Attributes:
    - OV/VE: Out-of-View - the target moves out of the current view.
    - OC: Occlusion - the target is partially or heavily occluded.
    - FM: Fast Motion - the target moves quickly.
    - SV: Scale Variation - the scale of the bounding boxes over the frames vary significantly.
    - TC/IC: Thermal/Infrared Crossover - the target has a similar temperature to other objects or background surroundings.
    - DBC: Dynamic Background Clusters - there are dynamic changes (e.g., waves, leaves, birds) in the background around the target.
    - LR: Low Resolution - the area of the bounding box is small.
    - TS: Target Scale - the target is with a tiny, small, medium or large scale.

    Args:
    path (str): The directory path containing the attribute JSON files.
    remote_client (object, optional): A client for remote file access. If None, local file system is used.
    
    Returns:
    dict: A dictionary where keys are label names (without file extension) and values are the loaded JSON data.
    """
    temp = {}

    if remote_client:
        dirs = remote_client.listdir(path)
    else:   
        dirs = os.listdir(path)

    for label in dirs:
        temp[label.split('.')[0]] = load_json(
            os.path.join(path, label), 
            remote_client=remote_client
        )

    return temp


def connect_sftp():
    """
    Establish an SFTP connection using the provided credentials.

    Args:
    host (str): The hostname or IP address of the SFTP server.
    port (int): The port number for the SFTP connection.
    user (str): The username for authentication.
    passw (str): The password for authentication.

    Returns:
    paramiko.SSHClient: An established SSH client with SFTP capabilities.
    """
    load_dotenv()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.getenv('SFTP_HOST'),
        port=os.getenv('SFTP_PORT'),
        username=os.getenv('SFTP_USERNAME'),
        password=os.getenv('SFTP_PASSWORD')
    )
    sftp_client = client.open_sftp()

    return sftp_client


def custom_collate_fn(batch):
    images = []
    bboxes = []

    for item in batch:
        if item['bbox'].numel() == 0:  # Check if bbox is not empty
            continue
        
        images.append(item['image'])
        bboxes.append(item['bbox'])

    # Convert lists to tensors
    images = torch.stack(images)
    bboxes = torch.stack(bboxes)

    return BatchData(image=images, bbox=bboxes)


def create_dataloader(dir_path, batch_size, shuffle=False, tsfm=None, remote=None, workers=4, mosaic=False, img_size=(640, 640), seed=11):
    """
    Create a DataLoader for the AntiUAVDataset.

    Args:
        dir_path (str): The root directory path for the dataset.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        tsfm (callable, optional): A function/transform to apply to the image samples. Defaults to None.
        remote (object, optional): A client for remote file access. If None, local file system is used. Defaults to None.
        workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        mosaic (bool, optional): Whether to use mosaic augmentation. Defaults to False.
        img_size (tuple, optional): Target size (height, width) to resize images to. Defaults to (640, 640).
        seed (int, optional): Random seed for shuffling the dataset. Defaults to 11.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the AntiUAVDataset.
    """
    from .AntiUAVDataset import AntiUAVDataset
    dataset = AntiUAVDataset(root_dir=dir_path, transform=tsfm, remote=remote, mosaic=mosaic, img_size=img_size, seed=seed)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=custom_collate_fn)

    return dataloader



def plot_sample_data(dataloader):
    """
    Plot a sample image from the given dataloader.

    Args:
        dataloader (DataLoader): The dataloader containing the dataset.
    """
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    # Sample 4 random images from the dataloader
    for i, sample in enumerate(dataloader):
        if i >= 4: 
            break
        
        image = sample.image[0].permute(1, 2, 0).numpy()
        bboxes = sample.bbox.numpy()

        # Draw the image
        axes[i].imshow(image)


        # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
        if len(bboxes[0]) > 0:
            bboxes = bboxes.squeeze(0)

        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Draw the bounding box
            rect = plt.Rectangle((int(x), int(y)), int(w), int(h), fill=False, edgecolor='cyan', linewidth=3)
            axes[i].add_patch(rect)

        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def create_mosaic_4_img(images, bboxes, target_size=(640, 640)):
    """
    Create a mosaic image from 4 input images and their corresponding bounding boxes.

    Args:
        images (list of np.ndarray): List of 4 images to be combined into a mosaic.
        bboxes (list of list of float): List of bounding boxes corresponding to each image. 
                                        Each bounding box should be in the format [x1, y1, x2, y2].
        target_size (tuple of int, optional): The size of the output mosaic image. Default is (640, 640).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The mosaic image.
            - torch.Tensor: The updated bounding boxes after resizing and positioning in the mosaic.

    Raises:
        ValueError: If the number of images or bounding boxes is less than 4 or if they do not match.
    """
    if len(images) < 4 or len(images) != len(bboxes):
        raise ValueError("Need at least 4 images and 4 sets of bounding boxes to create a mosaic.")
    
    # Create a new blank image
    mosaic = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate the size of each image in the mosaic
    img_width, img_height = target_size[0] // 2, target_size[1] // 2
    
    # Initialize list to store updated bounding boxes
    updated_bboxes = []
    
    # Resize and paste each image into the mosaic
    i = 0
    for img, bbox in zip(images, bboxes):
        # Get original image size and calculate position
        original_size = img.shape[:2]
        x = (i % 2) * img_width
        y = (i // 2) * img_height

        # Update bounding box coordinates
        scale_x = img_width / original_size[1]
        scale_y = img_height / original_size[0]
        
        x1, y1, x2, y2 = bbox
        x1 = x + (x1 * scale_x)
        y1 = y + (y1 * scale_y)
        x2 = x + (x2 * scale_x)
        y2 = y + (y2 * scale_y)

        if x1 >= x2 or y1 >= y2:
            continue

        updated_bboxes.append(torch.tensor([x1, y1, x2, y2]))
    
        # Resize and paste the image to placeholder
        img_resized = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
        mosaic[y:y+img_height, x:x+img_width] = img_resized

        if len(updated_bboxes) >= 4:
            updated_bboxes = torch.stack(updated_bboxes)
            return mosaic, updated_bboxes
            
        i += 1
        

def load_dataloader(train_path: str, val_path: str):
    """
    Load saved train and validation dataloaders from a pickle file using joblib

    Args:
        train_path: Path to the pickle file containing saved train dataloader
        val_path: Path to the pickle file containing saved validation dataloader
        
    Returns:
        Tuple containing (train_loader, val_loader)
    """
    train_loader = joblib.load(train_path)
    val_loader = joblib.load(val_path)
    
    return train_loader, val_loader
