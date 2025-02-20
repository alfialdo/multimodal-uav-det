import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from dvclive.lightning import DVCLiveLogger
import dvc.api as dvc
from dvclive import Live
from omegaconf import OmegaConf

from model import BaselineModel, DySOEM_SimFPN, DyYOLO
from dataset import load_dataloader

def train(config, train_loader, val_loader):
    # Load config
    trainer_cfg = config.train.trainer
    hparams= config.model.hparams
    model_name = config.model.name
    ckpt_cfg = config.train.checkpoint

    # Initialize model & dataloader
    if model_name == "DySOEM_SimFPN":
        model = DySOEM_SimFPN(hparams=hparams)
    elif model_name == "baseline":
        model = BaselineModel(hparams=hparams)
    elif model_name == "DyYOLO":
        model = DyYOLO(hparams=hparams)
    else:
        raise ValueError(f"Model {model} not supported")
    
    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename='best-{epoch:02d}-{val_loss:.4f}',
        dirpath=ckpt_cfg.dir,
        monitor=ckpt_cfg.monitor,
        mode=ckpt_cfg.mode,
        save_last=True
    )
    
    # Initialize trainer
    with Live(dvcyaml=False) as live:
        trainer = pl.Trainer(
            logger=DVCLiveLogger(log_model=False, experiment=live),
            callbacks=[checkpoint_callback],
            max_epochs=trainer_cfg.epochs,
            accelerator=trainer_cfg.accelerator,
            devices=trainer_cfg.devices,
            profiler=trainer_cfg.profiler,
            accumulate_grad_batches=trainer_cfg.grad_batches,
            limit_train_batches=trainer_cfg.train_batches,
            limit_val_batches=trainer_cfg.val_batches,
            val_check_interval=trainer_cfg.val_check_interval,
            gradient_clip_val=trainer_cfg.grad_clip_val,
            precision=trainer_cfg.precision,
            check_val_every_n_epoch=1
        )

        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    config = OmegaConf.load("params.yaml")

    if config.train.seed:
        seed_everything(config.train.seed, workers=True)

    # Set CUDA GPU matmul precision
    torch.set_float32_matmul_precision('high')

    train_loader, val_loader = load_dataloader(
        config.dataset.train_loader_path,
        config.dataset.val_loader_path
    )

    train(config, train_loader, val_loader)