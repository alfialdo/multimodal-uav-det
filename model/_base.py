import torch
import pytorch_lightning as pl
from torch import nn
from typing import List
import einops
from torchvision.ops import box_convert

from utils.datatype import BatchData, DetectionResults
from utils.metrics import filter_high_iou_bboxes, bbox_loss, objectness_loss, calculate_ap


class ConvModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False, activation='silu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True) if activation == 'silu' else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ObjectnessHead(pl.LightningModule):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        predict_c = n_anchors * 1
        self.n_anchors = n_anchors
        self.conv_bbox = nn.Conv2d(in_channels, predict_c, kernel_size=(1,1), stride=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_bbox(x)

        x = einops.rearrange(
            x, 'b (n_anchors obj) h w -> b n_anchors h w obj', 
            n_anchors=self.n_anchors, obj=1
        )

        x = self.sigmoid(x)

        return x
    

class BBoxHead(pl.LightningModule):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        predict_c = n_anchors * 4
        self.n_anchors = n_anchors
        self.conv_obj = nn.Conv2d(in_channels, predict_c, kernel_size=(1,1), stride=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_obj(x)
        x = einops.rearrange(
            x, 'b (n_anchors bbox) h w -> b n_anchors h w bbox', 
            n_anchors=self.n_anchors, bbox=4
        )

        x = self.sigmoid(x)

        return x

class YOLOHead(pl.LightningModule)  :
    def __init__(self, x_channels:List[int], anchors, input_size):
        super().__init__()
        # x_channels --> [x0_scale, x1_scale, x2_scale]
        self.input_scale = input_size[-1]
        self.anchors = torch.tensor(anchors)
        self.detection_head = nn.ModuleList()
        n_anchors = len(anchors[0])

        for x_in_channel in x_channels:
            self.detection_head.append(nn.ModuleDict(dict(
                obj=ObjectnessHead(x_in_channel, n_anchors),
                bbox=BBoxHead(x_in_channel, n_anchors)
            )))

    def forward(self, f_maps:List[torch.Tensor]):
        outs = []

        for head_idx, (f_map, det_head) in enumerate(zip(f_maps, self.detection_head)):
            scale = einops.parse_shape(f_map, 'b n_anchors h w')
            obj = det_head['obj'](f_map)
            bbox = det_head['bbox'](f_map)

            # Calculate bbox coordinates relative to grid
            bbox = self.__scale_bbox_size(
                scale['b'], scale['h'], scale['w'], 
                bbox, self.anchors[head_idx]
            )

            outs.append(DetectionResults(obj=obj, bbox=bbox))
        
        return outs
    
    
    def compute_metrics(self, outs:List[DetectionResults], batch:BatchData, head_scales:List[int], return_ap=False):
        device = outs[0].bbox.device
        batch_size = len(batch.bbox)
        total_loss = torch.tensor(0.0).to(device)
        total_ap = torch.tensor(0.0).to(device)

        # Loop through each batch
        for i in range(batch_size):
            # Loop through each detection head
            for head_idx in range(len(outs)):
                pred_bboxes = outs[head_idx].bbox[i] # head x, batch y
                pred_objs = outs[head_idx].obj[i]
                target_bboxes = batch.bbox[i]

                # Prepare preds and targets 
                p_bbox, p_obj = self.__prepare_preds(pred_bboxes, pred_objs)
                t_bbox = self.__scale_targets(target_bboxes, head_scales[head_idx], device)

                # Filter high IoU bboxes
                filtered_p_bbox, filtered_p_obj, t_obj = filter_high_iou_bboxes(p_bbox, p_obj, t_bbox)

                # Calculate loss
                total_loss += (bbox_loss(filtered_p_bbox, t_bbox) + objectness_loss(p_obj, t_obj))
                
                if return_ap:
                    map = calculate_ap(filtered_p_bbox, filtered_p_obj, t_bbox)['map']
                    total_ap += (map / len(outs))

        return total_loss / batch_size, total_ap / batch_size
    
    def __scale_targets(self, targets, scale_factor, device):
        # targets = box_convert(targets, in_fmt='xyxy', out_fmt='cxcywh')
        scaled_targets = targets / scale_factor
        return scaled_targets.to(device)
    
    def __prepare_preds(self, pred_bboxes, pred_objs):
        p_bbox = einops.rearrange(pred_bboxes, 'n_anchors h w bbox -> (n_anchors h w) bbox')
        p_obj = einops.rearrange(pred_objs, 'n_anchors h w obj -> (n_anchors h w) obj')

        p_bbox = box_convert(p_bbox, in_fmt='cxcywh', out_fmt='xyxy')
        p_obj = p_obj.squeeze()

        return p_bbox, p_obj

    def __scale_bbox_size(self, batch_size, h, w, bbox, anchors):
        # Setup grid coordnates for offsets
        device = bbox.device
        grid_x = torch.arange(w).repeat(batch_size, len(anchors), w, 1).to(device)
        grid_y = torch.arange(h).repeat(batch_size, len(anchors), h, 1).transpose(2, 3).to(device)

        # Setup anchor size
        anchor_w = einops.rearrange(anchors[:, 0], 'size -> 1 size 1 1').to(device)
        anchor_h = einops.rearrange(anchors[:, 1], 'size -> 1 size 1 1').to(device)

        # Calculate bbox coordinates relative to grid
        px = (bbox[..., 0] * 2 - 0.5 + grid_x)
        py = (bbox[..., 1] * 2 - 0.5 + grid_y)
        pw = (bbox[..., 2] * 2) ** 2 * anchor_w
        ph = (bbox[..., 3] * 2) ** 2 * anchor_h
        bbox = torch.stack([px, py, pw, ph], dim=-1)
        
        return bbox


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.learning_rate = hparams.lr
        self.optimizer = hparams.optim
        self.head_scales = hparams.head_scales

        self.backbone = None

        self.neck = None

        self.head = None
    
    def forward(self, x):
        return x
    
    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

        return optimizer    
    
    def training_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss, _ = self.head.compute_metrics(outs, batch, self.head_scales)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss, _ = self.head.compute_metrics(outs, batch, self.head_scales)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        # self.log('val_AP', ap, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return loss
    
    