import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert

from PIL import Image
import pandas as pd
import numpy as np
import os
import io
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ._helper import load_json, load_attributes, connect_sftp, create_mosaic_4_img, calculate_anchor_iou
from utils.test import generate_yolo_bboxes_test

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
            if isinstance(self.transform, A.Compose):
                tsfm = self.transform
            else:
                tsfm = A.Compose([
                    A.Resize(self.input_size, self.input_size),
                    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), shear=(-15, 15), p=1),
                    A.ToFloat(max_value=255.0), 
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))       

            results = tsfm(image=img, bboxes=bboxes, labels=labels)
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

    def __generate_yolo_bboxes(self, bbox:torch.Tensor):
        if bbox.numel() == 0:
            return []
        
        assert bbox.dim() == 2 and bbox.size(1) == 4, f"Expected bbox shape (N,4), got {bbox.shape}"

        # Normalize bboxes to value [0,1] with format cxcywh
        bbox = box_convert(bbox, 'xyxy', 'cxcywh')
        bbox /= self.input_size
        cx, cy, w, h = bbox.squeeze()

        # Assign anchors for each scale
        scale_bboxes = [torch.zeros(len(self.anchors[0]), s, s, 5) for s in self.head_size] # Create target in scaled space --> (n_anchors, grid_y, grid_x, [obj, cx, cy, w, h])

        for head_idx, size in enumerate(self.head_size):

            # Assign grid cell cx and cy
            grid_cx, grid_cy = (cx * size), (cy * size) 
            grid_x, grid_y = int(grid_cx), int(grid_cy) # grid cell position
            offset_cx, offset_cy = grid_cx - grid_x, grid_cy - grid_y # value [0, 1] refer to offset from grid cell point

            # Assign grid width and height
            grid_w, grid_h = w * size, h * size # offsets coord value could be > 1.0
            grid_bbox = torch.tensor([offset_cx, offset_cy, grid_w, grid_h])
            best_anchors, ious = calculate_anchor_iou(w, h, self.anchors[head_idx])

            # Assign only the highest iou anchor
            if ious[0] < 0.5:
                anchor_idx = best_anchors[0]
                scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 0] = torch.tensor(1.0)
                scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 1:5] = grid_bbox
            else:
                for anchor_idx, iou in zip(best_anchors, ious):

                    obj = torch.tensor(1.0 if iou >= 0.5 else 0.0)

                    # Add objectness to grid space
                    scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 0] = obj # value 0.0 or 1.0

                    # Add coordinates to grid space
                    scale_bboxes[head_idx][anchor_idx, grid_y, grid_x, 1:5] = grid_bbox
        
        
        generate_yolo_bboxes_test(scale_bboxes, self.head_size)
        return scale_bboxes

8