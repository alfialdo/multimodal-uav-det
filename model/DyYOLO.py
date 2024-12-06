import pytorch_lightning as pl
from torch import nn
import torch

from ._base import BaseModel, YOLOHead, DyConvModule
from dataset._helper import BatchData

### BACKBONE ###

class CNNBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)  

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()

        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x

        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels):
        super(ScalePrediction, self).__init__()
        self.conv = CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# DyYOLO Model configuration
class DyYOLO(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.layers = nn.ModuleList()
        self.attn_temp = hparams.attn_temperature
        x_out_channels = []
        in_channels = 3

        for module in hparams.layer_config:

            if module[0] == "B":
                num_repeats = module[1]
                self.layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )

            elif module[0] == "S":
                self.layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ScalePrediction(in_channels // 2),
                ]
                x_out_channels.append(in_channels)
                in_channels = in_channels // 2


            elif module[0] == "U":
                self.layers.append(
                    nn.Upsample(scale_factor=2),
                )
                in_channels = in_channels * 3

            else:
                out_channels, kernel_size, stride = module
                self.layers.append(
                    DyConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
        
        self.yolo_head = YOLOHead(x_out_channels, hparams.anchors, hparams.loss_balancing, hparams.bbox_loss_fn)

    def forward(self, x):
        outs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outs.append(layer(x))
                continue
            
            if isinstance(layer, DyConvModule):
                x = layer(x, self.attn_temp)
            else:
                x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)


            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        
        return self.yolo_head(outs)
    
    
    def training_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss, _, bbox_loss, obj_loss = self.yolo_head.compute_metrics(outs, batch)

        self.log('train_loss', loss, prog_bar=True, batch_size=len(batch))
        self.log('train_bbox_loss', bbox_loss, prog_bar=True, batch_size=len(batch))
        self.log('train_obj_loss', obj_loss, prog_bar=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch:BatchData, batch_idx):
        outs = self.forward(batch.image)
        loss, ap, bbox_loss, obj_loss = self.yolo_head.compute_metrics(outs, batch, return_ap=False)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log('val_bbox_loss', bbox_loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log('val_obj_loss', obj_loss, on_epoch=True, prog_bar=True, batch_size=len(batch))
        # self.log('val_AP', ap, on_epoch=True, prog_bar=True, batch_size=len(batch))