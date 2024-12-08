import torch
import pytorch_lightning as pl
from torch import nn
from typing import List
import einops
from torchvision.ops import box_convert, nms
from torch.nn import functional as F

from utils.datatype import BatchData, DetectionResults
from utils.metrics import bbox_loss, objectness_loss, no_obj_loss, calculate_ap
from utils.postprocess import calculate_iou


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

class DyConvModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, num_dy_conv=4):
        super().__init__()
        self.num_dy_conv = num_dy_conv
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Attention for 2d input
        if in_channels == 3:
            hidden_channels = num_dy_conv
        else:
            hidden_channels = int(in_channels * 0.25) + 1

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_dy_conv, kernel_size=1, bias=True)
        )

        # Dynamic convolution for 2d input  
        self.weights = nn.Parameter(torch.randn(num_dy_conv, out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
        self.silu = nn.SiLU(inplace=True)

    
    def forward(self, x, attn_temp):
        
        x_shape = einops.parse_shape(x, 'b c h w')
        batch_size, in_channels = x_shape['b'], x_shape['c']

        # Calculate attention scores
        attn_scores = self.attention(x)
        attn_scores = attn_scores.view(batch_size, -1)
        attn_scores = F.softmax(attn_scores / attn_temp, 1)

        # Aggregate weights
        weights = self.weights.view(self.num_dy_conv, -1)
        filters = torch.mm(attn_scores, weights)
        filters = einops.rearrange(
            filters, 'b (out_c in_c kh kw) -> (b out_c) in_c kh kw', 
            out_c=self.out_channels, in_c=in_channels, kh=self.kernel_size, kw=self.kernel_size
        )

        x = einops.rearrange(x, 'b c h w -> 1 (b c) h w')
        x = F.conv2d(x, filters, stride=self.stride, padding=self.padding, bias=None, groups=batch_size)
        x = einops.rearrange(x, '1 (b c) h w -> b c h w', b=batch_size)
        x = self.silu(self.bn(x))

        return x


class ObjectnessHead(pl.LightningModule):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        predict_c = n_anchors * 1
        self.n_anchors = n_anchors
        self.conv_obj = nn.Conv2d(in_channels, predict_c, kernel_size=(1,1), stride=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_obj(x)

        x = einops.rearrange(
            x, 'b (n_anchors obj) h w -> b n_anchors h w obj', 
            n_anchors=self.n_anchors, obj=1
        )

        # Apply sigmoid outside the model for flexibility
        # x = self.sigmoid(x)

        return x
    

class BBoxHead(pl.LightningModule):
    def __init__(self, in_channels, n_anchors):
        super().__init__()
        predict_c = n_anchors * 4
        self.n_anchors = n_anchors
        self.conv_bbox = nn.Conv2d(in_channels, predict_c, kernel_size=(1,1), stride=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_bbox(x)
        x = einops.rearrange(
            x, 'b (n_anchors bbox) h w -> b n_anchors h w bbox', 
            n_anchors=self.n_anchors, bbox=4
        )

        # Apply sigmoid outside the model for flexibility
        # x = self.sigmoid(x)

        return x

class YOLOHead(pl.LightningModule)  :
    def __init__(self, x_channels:List[int], anchors, head_scales, loss_balancing, bbox_loss_fn='mse'):
        super().__init__()
        # x_channels --> [x0_scale, x1_scale, x2_scale]
        self.anchors = torch.tensor(anchors).float()
        self.head_scales = torch.tensor(head_scales)
        self.detection_head = nn.ModuleList()
        n_anchors = len(anchors[0])

        self.obj_scales_w = loss_balancing.obj_scales_w
        self.bbox_w = loss_balancing.bbox_w
        self.objectness_w = loss_balancing.objectness_w
        self.no_obj_w = loss_balancing.no_obj_w
        
        self.bbox_loss_fn = bbox_loss_fn

        for x_in_channel in x_channels:
            self.detection_head.append(nn.ModuleDict(dict(
                obj=ObjectnessHead(x_in_channel, n_anchors),
                bbox=BBoxHead(x_in_channel, n_anchors)
            )))

    def forward(self, f_maps:List[torch.Tensor]):
        outs = []

        for f_map, det_head in zip(f_maps, self.detection_head):
            obj = det_head['obj'](f_map)
            bbox = det_head['bbox'](f_map)

            outs.append(DetectionResults(obj=obj, bbox=bbox))
        
        return outs
    
    def compute_metrics(self, outs:List[DetectionResults], batch:BatchData, return_ap=False):
        device = batch.image.device 
        batch_size = len(batch.image)
        bbox_losses = torch.tensor(0.0, device=device)
        obj_losses = torch.tensor(0.0, device=device)
        total_ap = torch.tensor(0.0, device=device)

        # Loop through each batch
        for i in range(batch_size):
            batch_pred_bbox = []
            batch_pred_obj = []

            # Loop through each detection head
            for head_idx in range(len(outs)):
                # scale anchors to head grid space
                scaled_anchors = self.anchors[head_idx] / self.head_scales[head_idx]

                # Get predictions and targets for current sample
                p_bbox = outs[head_idx].bbox[i] # (n_anchors, h, w, 4)
                p_obj = outs[head_idx].obj[i] # (n_anchors, h, w, 1)
                assert not torch.isnan(p_bbox).any(), "p_bbox contains NaN values"
                assert not torch.isnan(p_obj).any(), "p_obj contains NaN values"
                
                targets = batch.bbox[i][head_idx] # [N, (gcx, gcy, gw, gh)] based on grid cell
                target_cell = targets[..., 0] == 1.0 # mask for grid cells with obj
                t_bbox = targets[..., 1:] # (N, 4)
                t_obj = targets[..., 0] # (N, 1)


                # Decode predictions to head grid space and build target
                p_bbox_decoded = self.__pred_bbox_decoding(p_bbox, scaled_anchors)
                ious = calculate_iou(p_bbox_decoded, t_bbox, scaled_anchors, mask=target_cell)
                t_bbox = self.__build_target_bbox(t_bbox, scaled_anchors)

                # Calculate losses
                bbox_losses += self.bbox_w * bbox_loss(p_bbox_decoded[target_cell], t_bbox[target_cell], scaled_anchors, bbox_loss_fn=self.bbox_loss_fn)
                obj_losses += self.objectness_w * objectness_loss(p_obj[target_cell], ious * t_obj[target_cell], self.obj_scales_w[head_idx])
                obj_losses += self.no_obj_w * no_obj_loss(p_obj[~target_cell], t_obj[~target_cell])

                # Calculate prediction average precision 
                if return_ap:
                    p_bbox_decoded, p_obj = self.__prepare_nms_preds(p_bbox_decoded, p_obj)
                    batch_pred_bbox.append(p_bbox_decoded)
                    batch_pred_obj.append(p_obj)

            if return_ap:
                batch_pred_bbox = torch.cat(batch_pred_bbox, dim=0)
                batch_pred_obj = torch.cat(batch_pred_obj, dim=0)
                nms_filter = nms(batch_pred_bbox, batch_pred_obj, 0.5)
                total_ap += calculate_ap(batch_pred_bbox[nms_filter], batch_pred_obj[nms_filter], t_bbox)['map']

        # Calculate final metrics
        bbox_losses /= batch_size
        obj_losses /= batch_size 
        total_loss = bbox_losses + obj_losses
        total_ap = total_ap / batch_size if return_ap else None

        return total_loss, total_ap, bbox_losses, obj_losses    

    def __pred_bbox_decoding(self, pred_bboxes, head_anchors):
        device = pred_bboxes.device

        # Convert offsets back to coordinates
        pcx = pred_bboxes[..., 0].sigmoid() * 2 - 0.5
        pcy = pred_bboxes[..., 1].sigmoid() * 2 - 0.5
        pw = (pred_bboxes[..., 2].sigmoid() * 2) ** 2
        ph = (pred_bboxes[..., 3].sigmoid() * 2) ** 2

        # Create grid coordinates
        if self.bbox_loss_fn == 'ciou':
            pred_shape = einops.parse_shape(pred_bboxes, 'n_anchors h w bbox')
            grid_x = torch.arange(pred_shape['w']).repeat(pred_shape['n_anchors'], pred_shape['h'], 1).to(device)
            grid_y = torch.arange(pred_shape['h']).repeat(pred_shape['n_anchors'], pred_shape['w'], 1).transpose(1, 2).to(device)
            
            # Get anchors_values
            anchor_w = einops.rearrange(head_anchors[:, 0], 'size -> size 1 1').to(device)
            anchor_h = einops.rearrange(head_anchors[:, 1], 'size -> size 1 1').to(device)

            # Add grid coordinates and anchors to get final coordinates
            pcx = pcx + grid_x
            pcy = pcy + grid_y
            pw = pw * anchor_w
            ph = ph * anchor_h

        pred_bbox_decoded = torch.stack([pcx, pcy, pw, ph], dim=-1)

        return pred_bbox_decoded
    
    def __prepare_nms_preds(self, pred_bboxes, pred_obj):
        pred_bboxes = einops.rearrange(pred_bboxes, 'n_anchors h w bbox -> (n_anchors h w) bbox')
        pred_obj = einops.rearrange(pred_obj, 'n_anchors h w obj -> (n_anchors h w obj)')
        pred_bboxes = box_convert(pred_bboxes, in_fmt='cxcywh', out_fmt='xyxy')

        return pred_bboxes, pred_obj
    
    def __build_target_bbox(self, target_bboxes, head_anchors):
        device = target_bboxes.device

        if self.bbox_loss_fn == 'mse':
            head_anchors = einops.rearrange(head_anchors, 'n_anchors wh -> n_anchors 1 1 wh').to(device)

            # normalize width and height to match prediction
            target_bboxes[..., 2:] = torch.sqrt((1e-16 + target_bboxes[..., 2:]) / head_anchors) / 2

        elif self.bbox_loss_fn == 'ciou':
            # Create grid coordinates
            target_shape = einops.parse_shape(target_bboxes, 'n_anchors h w bbox')
            grid_x = torch.arange(target_shape['w']).repeat(target_shape['n_anchors'], target_shape['h'], 1).to(device)
            grid_y = torch.arange(target_shape['h']).repeat(target_shape['n_anchors'], target_shape['w'], 1).transpose(1, 2).to(device)
            
            # Add grid coordinates to get final coordinates
            target_bboxes[..., 0] = target_bboxes[..., 0] + grid_x  # Add grid x to cx
            target_bboxes[..., 1] = target_bboxes[..., 1] + grid_y  # Add grid y to cy
            

        return target_bboxes


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.learning_rate = hparams.lr
        self.optimizer = hparams.optim
        self.head_scales = hparams.head_scales
        self.lr_scheduler = hparams.lr_scheduler

        self.backbone = None

        self.neck = None

        self.head = None
    
    def forward(self, x):
        return x
    
    def configure_optimizers(self):
        if self.optimizer.name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.optimizer.momentum)
        elif self.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate / 10,
                max_lr=self.learning_rate,
                step_size_up=4000,
                mode='triangular2',
                cycle_momentum=False
            )

            return dict(optimizer=optimizer, lr_scheduler=scheduler)

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
    
    