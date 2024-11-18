import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import seed_everything

import os
import joblib
import dvc.api as dvc

from dataset import create_dataloader
from utils.datatype import Config

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
        print('Created test data loader..')

        return test_loader

    train_loader = create_dataloader(
        dir_path=os.path.join(config.root_dir, "train"),
        tsfm=train_tsfm,
        mosaic=config.mosaic,
        batch_size=config.train_batch_size,
        shuffle=True,
        **common_args
    )
    print('Created train data loader..')
    
    val_loader = create_dataloader(
        dir_path=os.path.join(config.root_dir, "val"),
        tsfm=val_tsfm,
        batch_size=config.val_batch_size,
        shuffle=True,
        **common_args
    )
    print('Created validation data loader..')

    return train_loader, val_loader

if __name__ == "__main__":
    config = Config(dvc.params_show())
    
    if config.train.seed:
        seed_everything(config.train.seed, workers=True)
    
    train_loader, val_loader = get_dataloader(config.dataset)

    joblib.dump(train_loader, config.dataset.train_loader_path)
    joblib.dump(val_loader, config.dataset.val_loader_path)