import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from torchvision.models import 
import torchvision.models as models
import torchmetrics 
from ..models.landmark import FaceLandmarkNet, quantized_mobilenet_v3, NaimishNet

class FaceLandmarkTask(pl.LightningModule):
    def __init__(self, pretrained=True, num_pts=136, lr=0.001, **kwargs):
        super().__init__()
        
        self.model: FaceLandmarkNet = NaimishNet(num_pts=num_pts)
        
        self.trn_loss: torchmetrics.MeanSquaredError = torchmetrics.MeanSquaredError()
        self.val_loss: torchmetrics.MeanSquaredError = torchmetrics.MeanSquaredError()
        
        self.trn_mae: torchmetrics.AverageMeter  = torchmetrics.MeanAbsoluteError()
        self.val_mae: torchmetrics.AverageMeter  = torchmetrics.MeanAbsoluteError()
        
        self.learning_rate = lr
        self.criterion = nn.L1Loss()
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        return optimizer
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images = batch['image']
        key_pts = batch['keypoints']
        
        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)
        
        # convert variables to floats for regression loss
        key_pts = key_pts.type(torch.FloatTensor).to(self.device)
        images = images.type(torch.FloatTensor).to(self.device)
        
        preds = self.model(images) # align with Attention.forward
        loss = self.criterion(preds, key_pts)
        return loss, preds, key_pts
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        trn_mae = self.trn_mae(preds, labels)
        trn_loss = self.trn_loss(preds, labels)
        
        self.log('trn_loss', trn_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True) 
        self.log('trn_mae', trn_mae,  prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss
        
    # def training_epoch_end(self, outs):
    #     self.log('trn_epoch_loss', self.trn_loss.compute(), logger=True)
    #     self.log('trn_epoch_avg', self.trn_avg.compute(), logger=True)
        
        
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        val_loss = self.val_loss(preds, labels)
        val_mae = self.val_mae(preds, labels)
        
        self.log('val_loss', val_loss, prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        self.log('val_mae', val_mae,  prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        
        return loss
    
    # def validation_epoch_end(self, outs):
    #     self.log('val_epoch_loss', self.val_loss.compute(), logger=True)
    #     self.log('val_epoch_avg', self.val_avg.compute(), logger=True)
    
    
    
if __name__ == '__main__':
    import sys
    import os
    import torch

    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, current_dir)
    mvn2 = FaceLandmarkTask()
    batch = torch.rand(2,3,224,224)
    result = mvn2(batch)
    print(result)