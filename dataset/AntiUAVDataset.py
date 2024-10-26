import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import io

from ._helper import load_json, load_attributes, connect_sftp
from utils.datatype import BatchData

# TODO: to apply Mosaics and Random Affine transform for image and bbox
class AntiUAVDataset(Dataset):
    def __init__(self, root_dir, transform=None, remote=None, size=None):
        self.data_set = os.path.basename(root_dir)
        self.remote = remote
        self.data = self.__load_data(root_dir)
        self.transform = transform
        self.size = size
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = self.__load_image(row.img_path)
        
        # Apply necessary transforms to the image
        if self.transform:
            bbox = self.__resize_bbox(img.size, row.gt_rect)
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return BatchData(image=img, bbox=bbox, obj=torch.tensor(row.exist))

    def __load_image(self, img_path):

        if self.remote:
            with connect_sftp() as sftp:
                with sftp.open(img_path, 'rb') as f:
                    f.prefetch()
                    img = f.read()
                    img = Image.open(io.BytesIO(img))
                    img.load()
        else:
            img = Image.open(img_path)

        return img
        
    
    def __load_data(self, root_dir):
        df = []

        if self.remote:
            sftp = connect_sftp()
            list_dir = sftp.listdir
        else:
            sftp = None
            list_dir = os.listdir

        attr_dir = os.path.join(os.path.dirname(root_dir), 'label_new')
        attr = load_attributes(attr_dir, remote_client=sftp)

        for seq in list_dir(root_dir):
            seq_dir = os.path.join(root_dir, seq)

            for cam_type in ['visible', 'infrared']:
                gt = load_json(os.path.join(seq_dir, f'{cam_type}.json'), remote_client=sftp)
                total_frame = len(gt['gt_rect'])

                img_dir = os.path.join(seq_dir, cam_type)
                img_paths = [os.path.join(img_dir, f"{cam_type}-{str(i).zfill(4)}.jpg") for i in range(total_frame)]
                
                df.append(pd.DataFrame(dict(
                    cam_type=[cam_type] * total_frame,
                    attribute=[attr[self.data_set][seq]] * total_frame,
                    img_path=img_paths,
                    gt_rect=gt['gt_rect'],
                    exist=gt['exist']
                )))
        
        if self.remote:
            sftp.close()

        # Filter images that not bounding box
        df = pd.concat(df, ignore_index=True)
        df = df[df['exist'] == 1].reset_index(drop=True)

        # Transform bbox to xyxy
        df['gt_rect'] = df.gt_rect.apply(lambda x: box_convert(torch.tensor(x), 'xywh', 'xyxy').squeeze(0))
        
        return df
    

    def __resize_bbox(self, img_size, bbox):
        w, h = img_size

        scale_x = self.size[0] / w
        scale_y = self.size[1] / h
        
        bbox[0] = int(bbox[0] * scale_x)
        bbox[1] = int(bbox[1] * scale_y)
        bbox[2] = int(bbox[2] * scale_x)
        bbox[3] = int(bbox[3] * scale_y)

        return bbox
    

        

