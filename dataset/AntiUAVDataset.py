import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from albumentations.pytorch import ToTensorV2

from PIL import Image
import pandas as pd
import numpy as np
import os
import io

from ._helper import load_json, load_attributes, connect_sftp, create_mosaic_4_img
from utils.datatype import BatchData

class AntiUAVDataset(Dataset):
    def __init__(self, root_dir, transform=None, remote=None, mosaic=False, img_size=(640, 640)):
        self.data_set = os.path.basename(root_dir)
        self.remote = remote
        self.data = self.__load_data(root_dir)
        self.transform = transform
        self.mosaic = mosaic
        self.img_size = img_size

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        if self.mosaic:
            rows = self.data.sample(4)
            images = [self.__load_image(x) for x in rows.img_path]
            bboxes = rows.gt_rect.tolist()
            img, bboxes = create_mosaic_4_img(images, bboxes, target_size=self.img_size)
            labels = [1] * len(bboxes)

        else:           
            row = self.data.iloc[idx]
            # TODO: check if we can use 1D for infrared images (grayscale=True)
            if row.cam_type == 'infrared':
                img = self.__load_image(row.img_path, grayscale=False)
            else:
                img = self.__load_image(row.img_path)

            bboxes = row.gt_rect.unsqueeze(0)
            labels = [1]

        # Apply necessary transforms to the image
        if self.transform:
            results = self.transform(image=img, bboxes=bboxes, labels=labels)
            img, bboxes = results['image'], torch.from_numpy(results['bboxes'])

        return dict(image=img, bbox=bboxes)

    def __load_image(self, img_path, grayscale=False):

        if self.remote:
            with connect_sftp() as sftp:
                with sftp.open(img_path, 'rb') as f:
                    f.prefetch()
                    img = f.read()
                    img = Image.open(io.BytesIO(img))
                    img.load()
        else:
            img = Image.open(img_path)

        if grayscale:
            img = img.convert('L')

        return np.array(img)
        
    
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

        # Filter images that not have bounding box
        df = pd.concat(df, ignore_index=True)
        df = df[df.exist == 1]
        df = df.loc[df.gt_rect.apply(lambda x: (x[2] > 0) and (x[3] > 0))].reset_index(drop=True)

        # Transform bbox to xyxy
        df['gt_rect'] = df.gt_rect.apply(lambda x: box_convert(torch.tensor(x), 'xywh', 'xyxy').squeeze(0))
        
        return df
    

        

