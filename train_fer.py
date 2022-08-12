from pathlib import Path
import logging
import argparse

import torch
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
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
    
    parser.add_argument('--freeze', type=str, default=None)
    parser.add_argument('--unfreeze', type=str, default=None)
    
    
    

    
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
    backbone_name = dict_args.get("backbone_name", "mobilenet_v3")
    
    datamod = FER2013DataModule(**dict_args)
    fertask = FERTask(**dict_args)
    
    freeze = dict_args.get("freeze", None)
    if freeze!=None:
        if freeze=="backbone":
            fertask.model.freeze_backbone()
        if freeze=="classifier":
            fertask.model.freeze_classifier()
        if freeze=="all":
            fertask.model.freeze_all()
            
    unfreeze = dict_args.get("unfreeze", None)
    if unfreeze!=None:
        if unfreeze=="backbone":
            fertask.model.unfreeze_backbone()
        if unfreeze=="classifier":
            fertask.model.unfreeze_classifier()
        if unfreeze=="all":
            fertask.model.unfreeze_all()
    
    
    
    # model_checkpoint = ModelCheckpoint(monitor="val_step_loss")
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints/',
        save_top_k=1,
        filename=backbone_name+"-emotion-{epoch:02d}-{val_loss:.4f}-{val_acc1:.4f}",
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    # seed_everything(0)
    earlystop = EarlyStopping(monitor="val_loss", patience=100, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger = TensorBoardLogger(save_dir='logs/', name='emotion')
    # swa = StochasticWeightAveraging(swa_lrs=1e-2)
    
    trainer = pl.Trainer.from_argparse_args(hparams, accumulate_grad_batches=7, callbacks=[model_checkpoint, earlystop, lr_monitor], logger=logger)
    trainer.fit(fertask, datamod)
    # with mlflow.start_run() as run:
    #     trainer.fit(facekeypoint, datamod)
    # trainer.save_checkpoint("checkpoints/latest.ckpt")
    
    
    metrics =  trainer.logged_metrics
    vacc, vloss, last_epoch = metrics['val_step_acc1'], metrics['val_step_loss'], trainer.current_epoch
    
    
    filename = f'emotion-{last_epoch:02d}_{vacc:.4f}_loss{vloss:.4f}.pth'
    saved_filename = str(Path('weights').joinpath(filename))
    
    logging.info(f"Prepare to save training results to path {saved_filename}")
    torch.save(fertask.model.state_dict(), saved_filename)