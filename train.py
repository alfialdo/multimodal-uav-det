from model.ProposedModel import ProposedModel
from dataset import create_dataloader

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2

import hydra
import os

def get_dataloader(config, test=False):

    common_args = dict(
        remote=config.remote,
        img_size=config.image_size,
    )

    train_tsfm = A.Compose([
        A.Resize(common_args['img_size'][0], common_args['img_size'][1]),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), shear=(-15, 15)),
        A.ToFloat(max_value=255.0), 
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    val_tsfm = A.Compose([
        A.Resize(common_args['img_size'][0], common_args['img_size'][1]),
        A.ToFloat(max_value=255.0), 
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    if test:
        test_loader = create_dataloader(
            dir_path=os.path.join(config.root_dir, "test"), 
            tsfm=val_tsfm,
            batch_size=config.val_batch_size,
            **common_args
        )

        return test_loader

    train_loader = create_dataloader(
        dir_path=os.path.join(config.root_dir, "train"),
        tsfm=train_tsfm,
        mosaic=config.mosaic,
        batch_size=config.train_batch_size,
        shuffle=True,
        **common_args
    )
    
    val_loader = create_dataloader(
        dir_path=os.path.join(config.root_dir, "val"),
        tsfm=val_tsfm,
        batch_size=config.val_batch_size,
        **common_args
    )

    return train_loader, val_loader


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(config):
    # Load config
    dataset_cfg = config.dataset
    trainer_cfg = config.train.trainer
    hparams= config.train.hparams
    model = config.train.model

    # Set weights precision
    torch.set_float32_matmul_precision(trainer_cfg.precision)

    # Set fix seed
    if trainer_cfg.seed:
        seed_everything(trainer_cfg.seed, workers=True)
    

    # Initialize model & dataloader
    if model == "proposed":
        logger = CSVLogger(save_dir="logs", name=model)
        model = ProposedModel(input_size=trainer_cfg.input_size, hparams=hparams)
    else:
        raise ValueError(f"Model {model} not supported")
    
    train_loader, val_loader = get_dataloader(dataset_cfg)

    # Intitialize trainer
    trainer = pl.Trainer(
        logger=logger,
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