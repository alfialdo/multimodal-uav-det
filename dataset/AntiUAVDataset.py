import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from albumentations.pytorch import ToTensorV2

from PIL import Image
import pandas as pd
import numpy as np
import os
import io

from ._helper import load_json, load_attributes, connect_sftp, create_mosaic_4_img, calculate_iou_wh
from utils.datatype import BatchData

class AntiUAVDataset(Dataset):
    def __init__(self, root_dir, config, transform=None, anchors=None, head_scales=None, seed=11):
        self.data_set = os.path.basename(root_dir)
        self.remote = config.remote
        self.seed = seed
        self.transform = transform
        self.mosaic = config.mosaic
        self.img_size = config.image_size
        self.input_size = config.image_size[0]
        self.format = config.format
        self.data = self.__load_data(root_dir)
        self.anchors = torch.tensor(anchors).float() / self.input_size # n anchors (w, h) per detection head
        self.head_size = torch.tensor([self.input_size // x for x in head_scales]) 
        self.format = config.format


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
        
        if self.format == 'yolo':
            bboxes = self.__generate_yolo_bboxes(bboxes)

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

        # Transform bbox based on format
        df['gt_rect'] = df.gt_rect.apply(lambda x: box_convert(torch.tensor(x), 'xywh', 'xyxy').squeeze(0))

        # Shuffle dataset
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df    

    def __generate_yolo_bboxes(self, bbox):
        if bbox.numel() == 0:
            print('generate yolo', bbox)
            return bbox

        # Normalize bboxes to value [0,1] with format cxcywh
        bbox = box_convert(bbox, 'xyxy', 'cxcywh')
        bbox /= self.input_size
        cx, cy, w, h = bbox.squeeze()

        # Assign anchors for each scale
        scale_bboxes = [torch.zeros(len(self.anchors[0]), s, s, 5) for s in self.head_size] # Create target in scaled space --> (n_anchors, grid_y, grid_x, [obj, cx, cy, w, h])

        for head_idx, size in enumerate(self.head_size):
            grid_x, grid_y = int(cx * size), int (cy * size)

            # Assign grid cell cx and cy
            grid_cx, grid_cy = (cx * size) - grid_x, (cy * size) - grid_y # value [0, 1]

            # Assign grid width and height
            grid_w, grid_h = w * size, h * size # value could be > 1.0
            grid_bbox = torch.tensor([grid_cx, grid_cy, grid_w, grid_h])
            
            for anchor_idx, anchors in enumerate(self.anchors[head_idx]):

                obj = 1.0 if calculate_iou_wh(w, h, anchors) >= 0.5 else 0.0

                # Add objectness to grid space
                scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 0] = obj # value 0.0 or 1.0

                # Add coordinates to grid space
                scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 1:5] = grid_bbox
        
        return scale_bboxes

