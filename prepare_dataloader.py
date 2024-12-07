import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import seed_everything

import os
import joblib
from omegaconf import OmegaConf

from dataset import create_dataloader

def get_dataloader(dataset_cfg, train_cfg, seed):
    img_w, img_h = dataset_cfg.image_size[0], dataset_cfg.image_size[1]
    common_args = dict(
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        seed=seed
    )

    # Create transform functions
    val_tsfm = A.Compose([
        A.Resize(img_w,img_h),
        # A.ToFloat(max_value=255.0), 
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


    # Create the data loaders
    train_loader = create_dataloader(
        dir_path=os.path.join(dataset_cfg.root_dir, "train"),
        shuffle=True,
        tsfm=True,
        **common_args
    )
    print('Created train data loader..')
    
    val_loader = create_dataloader(
        dir_path=os.path.join(dataset_cfg.root_dir, "val"),
        shuffle=False,
        tsfm=val_tsfm,
        **common_args
    )
    print('Created validation data loader..')

    test_loader = create_dataloader(
        dir_path=os.path.join(dataset_cfg.root_dir, "test"),
        shuffle=False,
        tsfm=val_tsfm,
        **common_args
    )
    print('Created test data loader..')

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    config = OmegaConf.load('params.yaml')
    seed = config.train.seed
    
    if seed:
        seed_everything(seed, workers=True)
    
    train_loader, val_loader, test_loader = get_dataloader(
        config.dataset,
        config.model.hparams,
        seed
    )

    joblib.dump(train_loader, config.dataset.train_loader_path)
    joblib.dump(val_loader, config.dataset.val_loader_path)
    joblib.dump(test_loader, config.dataset.test_loader_path)