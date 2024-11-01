import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.ops import box_iou, box_convert

from typing import List
import einops

from utils.datatype import DetectionResults, BatchData
from utils.metrics import bbox_loss, objectness_loss

# Custom Module
class ConvModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False, eps=1e-3, momentum=0.03, activation='silu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True) if activation == 'silu' else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class StemLayer(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvModule(in_channels, out_channels, kernel_size=(5,5), stride=(2,2), padding=(1,1), bias=False)
        
    
    def forward(self, x):
        x = self.conv(x)
        return x


# Backbone layer
class MDyConv(pl.LightningModule):
    def __init__(self, in_channels, attention_out_c, dy_kernel_size=3, dy_padding=1, dy_channel_size=None):
        super().__init__()
        if dy_channel_size:
            self.dy_channel_size = dy_channel_size
        else:
            self.dy_channel_size = in_channels

        self.dy_kernel_size = dy_kernel_size
        self.dy_padding = dy_padding

        self.base_conv = ConvModule(in_channels, self.dy_channel_size, kernel_size=(1,1), eps=1e-5, momentum=0.1, activation='relu')

        # Attention layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.dy_channel_size, attention_out_c, kernel_size=(1,1)),
            nn.ReLU(inplace=True)
        )


        self.channel_fc = nn.Conv2d(attention_out_c, self.dy_channel_size , kernel_size=(1,1))
        self.kernel_fc = nn.Conv2d(attention_out_c, int(self.dy_kernel_size ** 2), kernel_size=(1,1))
        
        # Activation function
        # self.relu = nn.ReLU()


    def forward(self, x):
        # First conv layer
        x = self.base_conv(x)
        residual = x

        # Attention mechanism for channel and kernel
        attn_x = self.attention(x)

        channel_w = self.channel_fc(attn_x)

        kernel_w = self.kernel_fc(attn_x)
        kernel_w = einops.rearrange(
            kernel_w, 'b (h w) 1 1 -> b 1 h w',
            h=self.dy_kernel_size, w=self.dy_kernel_size
        )

        # Convolve with first conv residual
        dy_conv = einops.rearrange(
            kernel_w * channel_w, 'b c h w -> (b c) 1 h w',
            h=self.dy_kernel_size, w=self.dy_kernel_size
        )

        # Perform dynamic convolution
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'b c h w -> 1 (b c) h w')
        x = F.conv2d(x, dy_conv, stride=(1,1), padding=self.dy_padding, groups=batch_size * self.dy_channel_size)   

        # Add residual
        x = einops.rearrange(x, '1 (b c) h w -> b c h w', b=batch_size)
        x = torch.add(x, residual)


        return x


class MDyCSPModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, reduction_ratio=2, dy_channel_size=None):
        super().__init__()
        
        base_out_c = in_channels * 2
        self.base_conv = ConvModule(in_channels, base_out_c, kernel_size=(3,3), stride=(2,2), padding=(1,1))

        self.conv1 = ConvModule(base_out_c, base_out_c // reduction_ratio, kernel_size=(1,1))
        self.conv2 = ConvModule(base_out_c, base_out_c // reduction_ratio, kernel_size=(1,1))

        if dy_channel_size:
            self.mdy_conv = MDyConv(base_out_c // reduction_ratio, 16, dy_kernel_size=3, dy_channel_size=dy_channel_size)
        else:
            self.mdy_conv = MDyConv(base_out_c // reduction_ratio, 16, dy_kernel_size=3)

        transition_c = base_out_c // reduction_ratio
        self.transition1 = ConvModule(128, transition_c, kernel_size=(1,1))
        self.transition2 = ConvModule(base_out_c, out_channels, kernel_size=(3,3), padding=(1,1))
    
    def forward(self, x):
        # Based layer
        x = self.base_conv(x)
        
        # Halfing layer for cross-stage
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # Computation layer
        x1 = self.mdy_conv(x1)

        # Cross-stage
        x1 = self.transition1(x1)

        x = torch.cat([x1,x2], dim=1)

        x = self.transition2(x)
        
        return x


# Neck layer
class MDyEncoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_in = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-5, affine=True)
        
        self.mdy_conv_1x1 = MDyConv(in_channels, 16, dy_kernel_size=1, dy_padding=0, dy_channel_size=in_channels // 3)
        self.mdy_conv_3x3 = MDyConv(in_channels, 16, dy_kernel_size=3, dy_padding=1, dy_channel_size=in_channels // 3)
        self.mdy_conv_5x5 = MDyConv(in_channels, 16, dy_kernel_size=5, dy_padding=2, dy_channel_size=in_channels // 3)
        
        self.group_norm_out = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-5, affine=True)

        # TODO: validate mlp conv layer position
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels , kernel_size=(1,1)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        )


    def forward(self, x):
        residual = x
        x = self.group_norm_in(x)

        x1 = self.mdy_conv_1x1(x)
        x2 = self.mdy_conv_3x3(x)
        x3 = self.mdy_conv_5x5(x)

        x = torch.cat([x1,x2,x3], dim=1)
        
        x = torch.add(x, residual)  
        residual = x

        x = self.group_norm_out(x)

        x = self.channel_mlp(x)

        # TODO: bug when adding with last residual
        # x = torch.add(x, residual)

        return x

class MFDFEncoderModule(pl.LightningModule):
    def __init__(self, x1_c_in, x2_c_in):
        super().__init__()
        encoder_x1_out_c = x1_c_in
        encoder_x2_out_c = x2_c_in

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(x2_c_in, x2_c_in // 4, kernel_size=(3,3), padding=(1,1))
        )
        self.downsample = nn.Conv2d(encoder_x1_out_c, encoder_x1_out_c, kernel_size=(3,3), stride=(2,2), padding=(1,1))

        self.encoder_x1 = MDyEncoder((x1_c_in // 2) * 3, encoder_x1_out_c)
        self.encoder_x2 = MDyEncoder((x2_c_in // 2) * 3, encoder_x2_out_c)


    def forward(self, x1, x2):
        f_map = self.upsample(x2)
        
        x1 = torch.cat([x1, f_map], dim=1)

        x1 = self.encoder_x1(x1)

        x = self.downsample(x1)
        
        x2 = torch.cat([x2, x], dim=1)
        
        x2 = self.encoder_x2(x2)

        return x1, x2

# Head layer
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
    

class RTMHead(pl.LightningModule):
    def __init__(self, x_c_in:list, anchors:list, det_scales:list):
        super().__init__()

        self.det_scales = det_scales
        self.detection_head = nn.ModuleList()
        self.anchors = anchors
        n_anchors = len(anchors[0])

        for in_channels in x_c_in:
            self.detection_head.append(nn.ModuleDict(dict(
                obj=ObjectnessHead(in_channels, n_anchors),
                bbox=BBoxHead(in_channels, n_anchors)
            )))

    
    def __calculate_bbox_size(self, batch_size, h, w, bbox, anchors):
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
    
    def forward(self, x1, x2):
        f_maps = [x1, x2]
        outs = []

        for head_idx, (f_map, det_head) in enumerate(zip(f_maps, self.detection_head)):
            scale = einops.parse_shape(f_map, 'b n_anchors h w')
            obj = det_head['obj'](f_map)
            bbox = det_head['bbox'](f_map)

            # Calculate bbox coordinates relative to grid
            bbox = self.__calculate_bbox_size(
                scale['b'], scale['h'], scale['w'], 
                bbox, self.anchors[head_idx]
            )

            outs.append(DetectionResults(obj=obj, bbox=bbox))
        
        return outs


class RTMUAVDet(pl.LightningModule):
    def __init__(self, input_size, anchors, learning_rate, optimizer='Adam', det_scales=[160, 80]):
        super().__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.input_size = input_size
        self.det_scales = det_scales

        self.backbone = nn.ModuleDict(dict(
            MDyCSP_1=nn.Sequential(
                # TODO: increase stem layer output channel?
                StemLayer(input_size[0], 32),
                MDyCSPModule(in_channels=32, out_channels=128, dy_channel_size=128),
            ),
            MDyCSP_2=MDyCSPModule(in_channels=128, out_channels=256)
        ))

        self.neck = MFDFEncoderModule(x1_c_in=128, x2_c_in=256)

        self.head = RTMHead(x_c_in=[128, 256], anchors=anchors, det_scales=det_scales)
    
    def forward(self, x):
        x1 = self.backbone['MDyCSP_1'](x)
        
        x2 = self.backbone['MDyCSP_2'](x1)
        
        x1, x2 = self.neck(x1, x2)
        
        outs = self.head(x1, x2)

        return outs
    
    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

        return optimizer

    def compute_loss(self, outs:List[DetectionResults], batch:BatchData):
        
        device = outs[0].bbox.device
        total_loss = torch.tensor(0.0).to(device)
        batch_size = len(batch.bbox)

        # Loop through each detection head
        for i in range(len(outs)):
            # TODO: must be calculated per batch size
            # Loop through each batch
            for pred_bboxes, pred_objs, target_bboxes in zip(outs[i].bbox, outs[i].obj, batch.bbox):
                # Prepare preds and targets
                p_bbox, p_obj = self.__prepare_preds(pred_bboxes, pred_objs)
                t_bbox = self.__scale_targets(target_bboxes, self.det_scales[i], device)

                # Filter high IoU bboxes
                filtered_p_bbox, t_obj = self.__filter_high_iou_bboxes(p_bbox, t_bbox)
                total_loss += (bbox_loss(filtered_p_bbox, t_bbox) + objectness_loss(p_obj, t_obj))

        return total_loss / batch_size

    def training_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss = self.compute_loss(outs, batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss = self.compute_loss(outs, batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return loss
    
    def __filter_high_iou_bboxes(self, preds, targets, iou_th=0.5):
        # Calculate IoU matrix
        device = preds.device
        ious = box_iou(preds, targets) 
        
        # Get maximum IoU for each prediction
        max_ious, _ = torch.max(ious, dim=1)  # (1D)

        # Filter IoU than less than threshold (0.5)
        filter_iou = max_ious > iou_th
        filtered_preds = preds[filter_iou, :]
        
        # Create objectness matrix (0 or 1)
        objectness = torch.zeros_like(max_ious).to(device)
        objectness[filter_iou] = 1.0
        
        return filtered_preds, objectness
    
    def __scale_targets(self, targets, det_scale, device):
        # targets = box_convert(targets, in_fmt='xyxy', out_fmt='cxcywh')
        scale_factor = self.input_size[1] // det_scale
        scaled_targets = targets / scale_factor

        return scaled_targets.to(device)
    
    def __prepare_preds(self, pred_bboxes, pred_objs):
        p_bbox = einops.rearrange(pred_bboxes, 'n_anchors h w bbox -> (n_anchors h w) bbox')
        p_obj = einops.rearrange(pred_objs, 'n_anchors h w obj -> (n_anchors h w) obj')

        p_bbox = box_convert(p_bbox, in_fmt='cxcywh', out_fmt='xyxy')

        return p_bbox, p_obj
