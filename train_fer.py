from pathlib import Path
import logging
import argparse

import torch
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from facelive.data.datamodule import IBug300WDataModule, IBugDlib300WDataModule, FER2013DataModule
from facelive.task.fer import FERTask
# import mlflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    # parser.add_argument('--data_type', type=str, default="std")
    parser.add_argument('--backbone_name', type=str, default="mobilenet_v3")
    parser.add_argument('--image_size', type=int, default=224)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    

    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    
    dict_args = vars(hparams)
    
    # mlflow.set_tracking_uri("http://localhost:54849")
    # mlflow.pytorch.autolog()
    # if dict_args["data_type"] == "dlib":
    #     datamod = IBugDlib300WDataModule(**dict_args)
    # elif dict_args["data_type"] == "std":
    #     datamod = IBug300WDataModule(**dict_args)
    # else:
    #     raise Exception("data_type must be either 'dlib' or 'std'")
    
    
    datamod = FER2013DataModule(**dict_args)
    fertask = FERTask(**dict_args)
    
    # model_checkpoint = ModelCheckpoint(monitor="val_step_loss")
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints/',
        save_top_k=1,
        filename="emotion-{epoch:02d}-{val_loss:.4f}-{val_acc1:.4f}",
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    # seed_everything(0)
    earlystop = EarlyStopping(monitor="val_loss",patience=10, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger = TensorBoardLogger(save_dir='logs/', name='facelandmark')
    
    trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[model_checkpoint, earlystop, lr_monitor], logger=logger)
    trainer.fit(fertask, datamod)
    # with mlflow.start_run() as run:
    #     trainer.fit(facekeypoint, datamod)
    # trainer.save_checkpoint("checkpoints/latest.ckpt")
    
    
    metrics =  trainer.logged_metrics
    vacc, vloss, last_epoch = metrics['val_step_avg'], metrics['val_step_loss'], trainer.current_epoch
    
    filename = f'facelandmark-e{last_epoch:02d}_avg{vacc:.4f}_loss{vloss:.4f}.pth'
    saved_filename = str(Path('weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(fertask.model.state_dict(), saved_filename)