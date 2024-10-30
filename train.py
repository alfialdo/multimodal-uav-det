from model.RTMUAVDet import RTMUAVDet
from dataset import create_dataloader

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2

import hydra
from omegaconf import OmegaConf

import os

def get_dataloader(**kwargs):
    root_dir = kwargs.get('root_dir')
    batch_size = kwargs.get('batch_size')
    remote = kwargs.get('remote')
    img_size = kwargs.get('img_size', (640, 640))
    test = kwargs.get('test', False)
    mosaic = kwargs.get('mosaic', False)

    tsfm = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), shear=(-15, 15)),
        A.ToFloat(max_value=255.0), 
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    if test:
        test_loader = create_dataloader(dir_path=os.path.join(root_dir, "test"), batch_size=batch_size, remote=remote, img_size=img_size)
        return test_loader
    
    train_loader = create_dataloader(dir_path=os.path.join(root_dir, "train"), tsfm=tsfm, batch_size=batch_size, remote=remote, img_size=img_size, mosaic=mosaic)
    val_loader = create_dataloader(dir_path=os.path.join(root_dir, "val"), tsfm=tsfm, batch_size=batch_size, remote=remote, img_size=img_size, mosaic=mosaic)

    return train_loader, val_loader


@hydra.main(config_path="conf", config_name="config")
def train(config):
    print("Training config:")
    print(OmegaConf.to_yaml(config))

    dataset_cfg = config.dataset
    trainer_cfg = config.train.trainer
    hparams= config.train.hparams

    if trainer_cfg.seed:
        seed_everything(trainer_cfg.seed, workers=True)
    
    train_loader, val_loader = get_dataloader(
        root_dir=dataset_cfg.root_dir,
        batch_size=dataset_cfg.batch_size,
        remote=dataset_cfg.remote,
        img_size=dataset_cfg.image_size,
        workers=dataset_cfg.workers,
        mosaic=dataset_cfg.mosaic
    )

    if config.train.model == "RTMUAVDet":
        anchors = torch.tensor(hparams.anchors)
        model = RTMUAVDet(input_size=trainer_cfg.input_size, anchors=anchors, learning_rate=hparams.lr, optimizer=hparams.optim, det_scales=hparams.det_scales)
    else:
        raise ValueError(f"Model {config.train.model} not supported")

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        profiler=trainer_cfg.profiler,
        accumulate_grad_batches=trainer_cfg.grad_batches,
        limit_train_batches=trainer_cfg.train_batches,
        limit_val_batches=trainer_cfg.val_batches
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()