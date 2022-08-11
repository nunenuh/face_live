from pathlib import Path
from tkinter import Image
from typing import *
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from .landmark import IBug300WDataset, IBugDLib300WDataset
from torchvision.datasets.fer2013 import FER2013
from torchvision import transforms
from .. import transforms as CT
from torchvision.datasets import ImageFolder


def landmark_transform_fn(size=96):
    tfm = transforms.Compose([
        CT.Rescale(250),
        CT.RandomCrop(size),
        CT.Normalize(),
        CT.ToTensor()
    ])
    
    return tfm

def fer2013_train_transform_fn(size=224):
    tmf = transforms.Compose([
            transforms.Resize(size),
            # transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    return tmf
    

def fer2013_valid_transform_fn(size=224):
    tmf = transforms.Compose([
            transforms.Resize(size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    return tmf


class IBug300WDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers=16,
                 image_size=96, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = IBug300WDataset(root=self.data_dir, train=True, transforms=landmark_transform_fn(size=self.image_size))
        self.validset = IBug300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn(size=self.image_size))
        self.predset = IBug300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn(size=self.image_size))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
class IBugDlib300WDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers: int = 16, 
                 image_size=96, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = IBugDLib300WDataset(root=self.data_dir, train=True, transforms=landmark_transform_fn(size=self.image_size))
        self.validset = IBugDLib300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn(size=self.image_size))
        self.predset = IBugDLib300WDataset(root=self.data_dir, train=False, transforms=landmark_transform_fn(size=self.image_size))

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
class FER2013DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = 'path/to/dir', batch_size: int = 32, num_workers: int = 16, 
                 image_size=48, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.traindir = str(Path(data_dir).joinpath('train'))
        self.validdir = str(Path(data_dir).joinpath('test'))
        
        self.image_size = image_size
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = ImageFolder(root=self.traindir, transform=fer2013_train_transform_fn(size=self.image_size))
        self.validset = ImageFolder(root=self.validdir, transform=fer2013_valid_transform_fn(size=self.image_size))
        self.predset = ImageFolder(root=self.validdir,  transform=fer2013_valid_transform_fn(size=self.image_size))
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)
        
        