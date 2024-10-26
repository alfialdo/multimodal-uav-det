import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from typing import List
import einops

from utils.datatype import DetectionResults, BatchData
from utils.metrics import iou_loss, cross_entropy_loss

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

    def forward(self, x):
        x = self.conv_bbox(x)

        x = einops.rearrange(
            x, 'b (n_anchors obj) h w -> b n_anchors h w obj', 
            n_anchors=self.n_anchors, obj=1
        )

        return x
    
class BBoxHead(pl.LightningModule):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        predict_c = n_anchors * 4
        self.n_anchors = n_anchors
        self.conv_obj = nn.Conv2d(in_channels, predict_c, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        x = self.conv_obj(x)
        x = einops.rearrange(
            x, 'b (n_anchors bbox) h w -> b n_anchors h w bbox', 
            n_anchors=self.n_anchors, bbox=4
        )

        return x
    

class RTMHead(pl.LightningModule):
    def __init__(self, x_c_in:list, n_anchors):
        super().__init__()

        self.detection_head = nn.ModuleList()

        for in_channels in x_c_in:
            self.detection_head.append(nn.ModuleDict(dict(
                obj=ObjectnessHead(in_channels, n_anchors),
                bbox=BBoxHead(in_channels, n_anchors)
            )))

    def forward(self, x1, x2):
        f_maps = [x1, x2]
        outs = []

        for f_map, det_head in zip(f_maps, self.detection_head):
            outs.append(DetectionResults(
                # Set output shape for prediction
                obj=einops.rearrange(det_head['obj'](f_map), 'b n_anchors h w obj -> b (n_anchors h w) obj'),
                bbox=einops.rearrange(det_head['bbox'](f_map), 'b n_anchors h w bbox -> b (n_anchors h w) bbox')
            ))
        
        return outs


class RTMUAVDet(pl.LightningModule):
    def __init__(self, img_channels, n_anchors, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.backbone = nn.ModuleDict(dict(
            MDyCSP_1=nn.Sequential(
                # TODO: increase stem layer output channel?
                StemLayer(img_channels, 32),
                MDyCSPModule(in_channels=32, out_channels=128, dy_channel_size=128),
            ),
            MDyCSP_2=MDyCSPModule(in_channels=128, out_channels=256)
        ))

        self.neck = MFDFEncoderModule(x1_c_in=128, x2_c_in=256)

        self.head = RTMHead(x_c_in=[128, 256], n_anchors=n_anchors)
    
    def forward(self, x):
        x1 = self.backbone['MDyCSP_1'](x)
        
        x2 = self.backbone['MDyCSP_2'](x1)
        
        x1, x2 = self.neck(x1, x2)
        
        outs = self.head(x1, x2)

        return outs
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, logits:List[DetectionResults], batch:BatchData):
        total_loss = 0

        for det in logits:
            total_loss += (iou_loss(det.bbox, batch.bbox) + cross_entropy_loss(det.obj, batch.obj))

        return (total_loss / len(logits))

    def training_step(self, batch:BatchData, batch_idx):
        logits = self.forward(batch.image)
        loss = self.compute_loss(logits, batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch:BatchData, batch_idx):
        logits = self.forward(batch.image)
        loss = self.compute_loss(logits, batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return loss
