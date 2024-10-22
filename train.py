from model.RTMUAVDet import RTMUAVDet
from dataset import create_dataloader

from torchvision import transforms as T
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import os

# TODO: add to config --> root_dir, batch_size, remote, img_size, tsfm (?)
def get_dataloader(**kwargs):
    root_dir = kwargs.get('root_dir')
    batch_size = kwargs.get('batch_size')
    remote = kwargs.get('remote')
    img_size = kwargs.get('img_size', (640, 640))
    test = kwargs.get('test', False)

    tsfm = T.Compose([
        T.Resize(size=img_size), # Resize to img_size
        T.ToTensor(),
        # T.Lambda(lambda x: x / 255.0)  # Scale pixel values to range 0-1
    ])

    if test:
        test_loader = create_dataloader(dir_path=os.path.join(root_dir, "test"), batch_size=batch_size, shuffle=False, remote=remote, img_size=img_size)
        return test_loader
    
    train_loader = create_dataloader(dir_path=os.path.join(root_dir, "train"), tsfm=tsfm, batch_size=batch_size, shuffle=True, remote=remote, img_size=img_size)
    val_loader = create_dataloader(dir_path=os.path.join(root_dir, "val"), tsfm=tsfm, batch_size=batch_size, shuffle=False, remote=remote, img_size=img_size)

    return train_loader, val_loader



if __name__ == "__main__":
    # TODO: add config --> root_dir, batch_size, remote, img_size, seed
    
    seed_everything(11, workers=True)
    
    train_loader, val_loader = get_dataloader(
        root_dir='/home/wicomai/dataset/Anti-UAV-RGBT',
        batch_size=8,
        remote=True,
        img_size=(640, 640)
    )

    # TODO: add to config --> model and trainer configuration
    model = RTMUAVDet(img_channels=3, n_anchors=3)
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='gpu',
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)