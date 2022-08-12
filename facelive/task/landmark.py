import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler

# from torchvision.models import 
import torchvision.models as models
import torchmetrics 
from .metrics import RootMeanSquaredError
from ..models.landmark import FaceLandmarkNet, quantized_mobilenet_v3, NaimishNet

class FaceLandmarkTask(pl.LightningModule):
    def __init__(self, pretrained=True, network_name="naimish", num_pts=136, lr=0.001, **kwargs):
        super().__init__()
        
        if network_name=="naimish":
            self.model: NaimishNet = NaimishNet(num_pts=num_pts)
        elif network_name=="landmark":
            self.model: FaceLandmarkNet = FaceLandmarkNet(backbone_name=kwargs.get("backbone_name", None), num_pts=num_pts)
            
        self.trn_mse: torchmetrics.MeanSquaredError = torchmetrics.MeanSquaredError()
        self.val_mse: torchmetrics.MeanSquaredError = torchmetrics.MeanSquaredError()
        
        self.trn_rmse: RootMeanSquaredError  = RootMeanSquaredError()
        self.val_rmse: RootMeanSquaredError  = RootMeanSquaredError()
        
        self.learning_rate = lr
        self.criterion = nn.SmoothL1Loss()
        
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-08)
        sch_cosine = lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0, last_epoch=-1, verbose=False)
        sch_cyclic = lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=0.1)
        return [optimizer], [sch_cosine, sch_cyclic]
    
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
        trn_rmse = self.trn_rmse(preds, labels)
        trn_mse = self.trn_mse(preds, labels)
        
        
        self.log('trn_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('trn_mse', trn_mse, prog_bar=True, logger=True, on_step=True, on_epoch=True) 
        self.log('trn_rmse', trn_rmse,  prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        return loss
        
    # def training_epoch_end(self, outs):
    #     self.log('trn_epoch_loss', self.trn_mse.compute(), logger=True)
    #     self.log('trn_epoch_avg', self.trn_avg.compute(), logger=True)
        
        
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        val_mse = self.val_mse(preds, labels)
        val_rmse = self.val_rmse(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_mse', val_mse, prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        self.log('val_rmse', val_rmse,  prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        
        return loss
    
    # def validation_epoch_end(self, outs):
    #     self.log('val_epoch_loss', self.val_mse.compute(), logger=True)
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