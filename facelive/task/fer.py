import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from torchvision.models import 
import torchvision.models as models
import torchmetrics 
# from .metrics import RootMeanSquaredError
# from ..models.landmark import FaceLandmarkNet, quantized_mobilenet_v3, NaimishNet
from ..models.fer import FERNet

class FERTask(pl.LightningModule):
    def __init__(self, pretrained=True, network_name="naimish", num_classes=7, lr=0.001, **kwargs):
        super().__init__()
        self.model: FERNet = FERNet(backbone_name=kwargs.get("backbone_name", None), num_classes=num_classes)
        
        self.trn_acc1: torchmetrics.Accuracy = torchmetrics.Accuracy(top_k=1)
        self.trn_acc5: torchmetrics.Accuracy = torchmetrics.Accuracy(top_k=5)
        
        self.val_acc1: torchmetrics.Accuracy = torchmetrics.Accuracy(top_k=1)
        self.val_acc5: torchmetrics.Accuracy = torchmetrics.Accuracy(top_k=5)
        
        self.learning_rate = lr
        self.criterion = nn.CrossEntropyLoss()
        
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)
        return optimizer
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        images, target = batch

        # convert variables to floats for regression loss
        images = images.to(self.device)
        target = target.to(self.device)
        
        preds = self.model(images) # align with Attention.forward
        loss = self.criterion(preds, target)
        return loss, preds, target
    
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        trn_acc1 = self.trn_acc1(preds, labels)
        trn_acc5 = self.trn_acc5(preds, labels)
        
        
        self.log('trn_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('trn_acc1', trn_acc1, prog_bar=True, logger=True, on_step=True, on_epoch=True) 
        self.log('trn_acc5', trn_acc5, prog_bar=True, logger=True, on_step=True, on_epoch=True) 
        
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, batch_idx)
        val_acc1 = self.val_acc1(preds, labels)
        val_acc5 = self.val_acc5(preds, labels)
        
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('val_acc1', val_acc1, prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        self.log('val_acc5', val_acc5, prog_bar=True, logger=True,  on_step=True, on_epoch=True) 
        
        
        return loss
    
    
    
if __name__ == '__main__':
    import sys
    import os
    import torch

    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, current_dir)
    mvn2 = FERTask()
    batch = torch.rand(2,3,224,224)
    result = mvn2(batch)
    print(result)