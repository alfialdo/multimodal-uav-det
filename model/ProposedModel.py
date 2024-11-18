import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.ops import box_convert

from typing import List
import einops

from utils.datatype import DetectionResults, BatchData
from utils.metrics import bbox_loss, objectness_loss, calculate_ap, filter_high_iou_bboxes
from ._base import BaseModel, ConvModule, YOLOHead


# Input Stem Layer 
class AdaptiveStemLayer(pl.LightningModule):
    def __init__(self, out_channels):
        super().__init__()
        self.gray_conv = ConvModule(1, out_channels, kernel_size=(1,1), bias=False, activation='silu')
        self.rgb_conv = ConvModule(3, out_channels, kernel_size=(1,1), bias=False, activation='silu')
        
    def forward(self, x):
        # TODO: check if we need to use dynamic channel?
        if x.size(1) == 1:
            x = self.gray_conv(x)
        else:
            x = self.rgb_conv(x)
        return x
    
class InputStemLayer(pl.LightningModule):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = ConvModule(3, out_channels, kernel_size=(1,1), bias=False, activation='silu')
        
    def forward(self, x):
        return self.conv(x)


### BACKBONE ###
# Create SOE Module - Small-object Enchanement Module
class DynamicSOEM(pl.LightningModule):
    def __init__(self, in_channels, num_dy_conv=3, dy_kernel_size=3, downsample_factor=2, reduction_ratio=2): 
        super().__init__()
        self.k = downsample_factor

        # Attention Module
        # TODO: validate num of hidden features
        in_attn = (downsample_factor ** 2) * in_channels
        hidden_features = max(1, in_attn//4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=in_attn, out_features=hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=num_dy_conv)
        )

        self.attn_softmax = nn.Softmax(dim=-1)

        # Dynamic conv
        self.dy_convs = nn.ModuleList([
            nn.Conv2d(in_attn, in_attn//reduction_ratio, kernel_size=dy_kernel_size, padding=dy_kernel_size//2, stride=1)
            for _ in range(num_dy_conv)
        ])

        self.bn = nn.BatchNorm2d(num_features= in_attn//reduction_ratio, affine=True)
        self.silu = nn.SiLU(inplace=True)


    def forward(self, x, attn_temp):
        f_map = []
        spatial_size = einops.parse_shape(x, 'b c h w')['w']

        # Downsample the feature map based on k factor
        for n in range(self.k ** 2):
            i, j = n // self.k, n % self.k
            f_map.append(x[..., i:spatial_size:self.k, j:spatial_size:self.k])

        f_map = torch.cat(f_map, dim=1)
    
        # Compute attention weights
        attn_weights = self.attention(f_map)
        attn_weights = self.attn_softmax(attn_weights / attn_temp)

        # Apply dynamic convolutions
        x = []
        for i, dy_conv in enumerate(self.dy_convs):
            # prepare weights for broadcasting
            w = einops.rearrange(attn_weights[:, i], 'n -> n 1 1 1')

            # scoring the convolution
            f_map_weighted = torch.multiply(w, dy_conv(f_map))
            x.append(f_map_weighted)
        
        x = torch.sum(torch.stack(x), dim=0)
        x = self.silu(self.bn(x))

        return x
        

### NECK ###
# Create Simplified FPN Module
class SimplifiedFPN(pl.LightningModule):
    def __init__(self, x_in_channels:List[int], conv_out_kernel=3):
        super().__init__()
        
        self.x2_in_down = nn.Conv2d(x_in_channels[2], x_in_channels[1], kernel_size=1, stride=1)
        self.center_down = nn.Conv2d(x_in_channels[1], x_in_channels[0], kernel_size=1, stride=1)

        self.x0_conv_out = ConvModule(x_in_channels[0], x_in_channels[0], kernel_size=conv_out_kernel, padding=conv_out_kernel//2, activation='silu')
        self.x1_conv_out = ConvModule(x_in_channels[1], x_in_channels[1], kernel_size=conv_out_kernel, padding=conv_out_kernel//2, activation='silu')
        self.x2_conv_out = ConvModule(x_in_channels[2], x_in_channels[2], kernel_size=conv_out_kernel, padding=conv_out_kernel//2, activation='silu')

        self.x0_out_up = nn.Conv2d(x_in_channels[0], x_in_channels[1], kernel_size=1, stride=2)
        self.x1_out_up = nn.Conv2d(x_in_channels[1], x_in_channels[2], kernel_size=1, stride=2)

    # x0:small, x1:medium, x2:large
    def forward(self, f_maps:List[torch.Tensor]):
        x0, x1, x2 = f_maps
        center_node = x1 + self.x2_in_down(F.interpolate(x2, scale_factor=2, mode='nearest')) + x1

        x0 = x0 + self.center_down(F.interpolate(center_node, scale_factor=2, mode='nearest'))
        x1 = center_node + self.x0_out_up(x0)
        x2 = x2 + self.x1_out_up(x1)
 
        x0 = self.x0_conv_out(x0)
        x1 = self.x1_conv_out(x1)
        x2 = self.x2_conv_out(x2)

        return x0, x1, x2
    

# Propose Model Configuration
class ProposedModel(BaseModel):
    def __init__(self, input_size, hparams, stem_out_channels=32):
        super().__init__(hparams)
        self.attn_temperature = hparams.attention_temperature
        self.input_stem = InputStemLayer(stem_out_channels)

        x_in_scales = [stem_out_channels, stem_out_channels * 2, stem_out_channels * 4]

        assert len(hparams.num_dy_conv) == len(hparams.dy_kernel_size), 'Num of dy_conv and dy_kernel_size must be the same'
        self.backbone = nn.ModuleList()
        for i, (n_dy_conv, k_size) in enumerate(zip(hparams.num_dy_conv, hparams.dy_kernel_size)):
            self.backbone.append(
                DynamicSOEM(in_channels=x_in_scales[i], num_dy_conv=n_dy_conv, dy_kernel_size=k_size)
            )
       
        x_out_channels = [x*2 for x in x_in_scales]
        self.neck = SimplifiedFPN(x_out_channels)
        self.yolo_head = YOLOHead(x_out_channels, hparams.anchors, input_size)

    def forward(self, x, attn_temp=1.0):
        x = self.input_stem(x)
        x_features = []

        for dy_soem in self.backbone:
            x = dy_soem(x, attn_temp)
            x_features.append(x)

        x0, x1, x2 = self.neck(x_features)
        del x_features

        outs = self.yolo_head([x0, x1, x2])
        del x0, x1, x2 

        return outs

    def training_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image, attn_temp=self.attn_temperature)
        loss, _ = self.yolo_head.compute_metrics(outs, batch, self.head_scales)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))

        return loss

    # TODO: add correct AP and AR metrics calculation here
    def validation_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss, _ = self.yolo_head.compute_metrics(outs, batch, self.head_scales)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        # self.log('val_AP', ap, on_epoch=True, prog_bar=True, batch_size=len(batch))

        return loss