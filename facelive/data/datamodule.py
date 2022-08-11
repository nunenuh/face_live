from typing import *
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from .landmark import IBug300WDataset, IBugDLib300WDataset
from torchvision.datasets.fer2013 import FER2013
from torchvision import transforms
from .. import transforms as CT


def landmark_transform_fn(size=96):
    tfm = transforms.Compose([
        CT.Rescale(250),
        CT.RandomCrop(size),
        CT.Normalize(),
        CT.ToTensor()
    ])
    
    return tfm


class IBug300WDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers=16, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = IBug300WDataset(root=self.data_dir, train=True, transforms=landmark_transform_fn())
        self.validset = IBug300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn())
        self.predset = IBug300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn())

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
class IBugDlib300WDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers: int = 16, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = IBugDLib300WDataset(root=self.data_dir, train=True, transforms=landmark_transform_fn())
        self.validset = IBugDLib300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn())
        self.predset = IBugDLib300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn())

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
class FER2013DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 32, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = FER2013(root=self.data_dir, split='train', transforms=None)
        self.validset = FER2013(root=self.data_dir, split="test", transforms=None)
        self.predset = FER2013(root=self.data_dir, split="test", transforms=None)
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)
        
        