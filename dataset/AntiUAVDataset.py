import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import io

from ._helper import load_json, load_attributes, connect_sftp


class AntiUAVDataset(Dataset):
    def __init__(self, root_dir, transform=None, remote=None, size=None):
        self.remote = remote

        # Initialize SFTP connection if using remote dataset
        if self.remote:
            self.client = connect_sftp(
                host=os.getenv('DS_HOSTNAME'),
                port=os.getenv('DS_PORT'),
                user=os.getenv('DS_USERNAME'),
                passw=os.getenv('DS_PASSWORD')
            )
            self.sftp = self.client.open_sftp()

        else:
            self.client = None
            self.sftp = None
        
        self.data = self.__load_data(root_dir)
        self.transform = transform
        self.size = size
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = self.__load_image(row.img_path)
        bbox = torch.tensor(row.gt_rect)
        
        # Apply necessary transforms to the image
        if self.transform:
            bbox = self.__resize_bbox(img.size, bbox)
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return dict(image=img, bbox=bbox, exist=torch.tensor(row.exist))
    
    def __del__(self):
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()


    def __load_image(self, img_path):

        if self.remote:
            with self.sftp.open(img_path, 'rb') as f:
                img = f.read()
                img = Image.open(io.BytesIO(img))
                img.load()

        else:
            img = Image.open(img_path)

        return img
        
    
    def __load_data(self, root_dir):
        attr_dir = os.path.join(os.path.dirname(root_dir), 'label_new')
        data_set = os.path.basename(root_dir)

        df = dict(
            cam_type=[],
            img_path=[],
            attribute=[],
            gt_rect=[],
            exist=[]
        )
        attr = load_attributes(attr_dir, remote_client=self.sftp)
        list_dir = self.sftp.listdir if self.remote else os.listdir

        for seq in list_dir(root_dir):
            seq_dir = os.path.join(root_dir, seq)

            for cam_type in ['visible', 'infrared']:
                labels = load_json(os.path.join(seq_dir, f'{cam_type}.json'), remote_client=self.sftp)
                img_dir = os.path.join(seq_dir, cam_type)
                img_paths = sorted(list_dir(img_dir))
                img_paths = [os.path.join(img_dir, x) for x in img_paths]

                df['cam_type'] += [cam_type] * len(img_paths)
                df['img_path'] += img_paths
                df['attribute'] += [attr[data_set][seq]] * len(img_paths)
                df['gt_rect'] += labels['gt_rect']
                df['exist'] += labels['exist']
                
        # Filter images that not bounding box
        df = pd.DataFrame(df)
        df = df[df['exist'] == 1].reset_index(drop=True)

        return df
    

    def __resize_bbox(self, img_size, bbox):
        bbox_xyxy = box_convert(bbox, 'xywh', 'xyxy').squeeze(0)
        w, h = img_size

        scale_x = self.size[0] / w
        scale_y = self.size[1] / h
        
        bbox_xyxy[0] = int(bbox_xyxy[0] * scale_x)
        bbox_xyxy[1] = int(bbox_xyxy[1] * scale_y)
        bbox_xyxy[2] = int(bbox_xyxy[2] * scale_x)
        bbox_xyxy[3] = int(bbox_xyxy[3] * scale_y)

        bbox_xywh = box_convert(bbox_xyxy, 'xyxy', 'xywh')

        
        return bbox_xywh
    

        

