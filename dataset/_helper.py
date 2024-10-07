import os
import json
import paramiko
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


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
        with remote_client.open(path, 'r') as f:
            data = json.load(f)
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


def connect_sftp(host, port, user, passw):
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
        hostname=host,
        port=port,
        username=user,
        password=passw
    )

    return client



def create_dataloader(dir_path, batch_size, shuffle=False, tsfm=None, remote=None, img_size=(640,640)):
    """
    Create a DataLoader for the AntiUAVDataset.

    Args:
    dir_path (str): The root directory path for the dataset.
    batch_size (int): The batch size for the DataLoader.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
    tsfm (callable, optional): A function/transform to apply to the image samples. Defaults to None.
    remote (object, optional): A client for remote file access. If None, local file system is used. Defaults to None.
    img_size (tuple, optional): The size to which images should be resized. Defaults to (640,640).

    Returns:
    torch.utils.data.DataLoader: A DataLoader for the AntiUAVDataset.
    """
    from .AntiUAVDataset import AntiUAVDataset
    dataset = AntiUAVDataset(root_dir=dir_path, transform=tsfm, remote=remote, size=img_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

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
    for i, ax in enumerate(axes):
        sample = next(iter(dataloader))
        image = sample['image'].squeeze(0).permute(1, 2, 0).numpy()
        bbox = sample['bbox'].squeeze(0).numpy()

        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = bbox

        # Draw the image
        ax.imshow(image)

        # Draw the bounding box
        rect = plt.Rectangle((int(x), int(y)), int(w), int(h), fill=False, edgecolor='cyan', linewidth=3)
        ax.add_patch(rect)

        ax.set_title(f"Sample {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    

