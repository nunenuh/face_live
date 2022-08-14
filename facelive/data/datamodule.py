from pathlib import Path
from tkinter import Image
from typing import *
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from .landmark import IBug300WDataset, IBugDLib300WDataset
from .fer import FER2013
from torchvision import transforms
from .. import transforms as CT
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

import torchlm


def landmark_transform_fn(size=96):
    tfm = transforms.Compose([
        CT.Rescale(250),
        CT.RandomCrop(size),
        CT.Normalize(),
        CT.ToTensor()
    ])
    return tfm

def train_landmarks_transform_fn(size=224, rotate=30):
    tfm = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMaskMixUp(prob=0.25),
        torchlm.LandmarksRandomBackgroundMixUp(prob=0.25),
        torchlm.LandmarksRandomScale(prob=0.25),
        torchlm.LandmarksRandomTranslate(prob=0.25),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.25),
        torchlm.LandmarksRandomBrightness(prob=0.25),
        torchlm.LandmarksRandomRotate(rotate, prob=0.25, bins=8),
        torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.25),
        torchlm.LandmarksResize((size, size), keep_aspect=False),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
    ])
    return tfm

def example_landmarks_transform_fn(size=224, rotate=30):
    tfm = torchlm.LandmarksCompose([
        torchlm.LandmarksRandomMaskMixUp(prob=0.25),
        torchlm.LandmarksRandomBackgroundMixUp(prob=0.25),
        torchlm.LandmarksRandomScale(prob=0.25),
        torchlm.LandmarksRandomTranslate(prob=0.25),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.25),
        torchlm.LandmarksRandomBrightness(prob=0.25),
        torchlm.LandmarksRandomRotate(rotate, prob=0.25, bins=8),
        torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.25),
        torchlm.LandmarksResize((size, size), keep_aspect=False),
        # torchlm.LandmarksNormalize(),
    ])
    return tfm

def valid_landmarks_transform_fn(size=224):
    tfm = torchlm.LandmarksCompose([
        torchlm.LandmarksResize((size, size), keep_aspect=False),
        torchlm.LandmarksNormalize(),
        torchlm.LandmarksToTensor(),
    ])
    
    return tfm

def fer2013_train_transform_fn(size=224):
    tmf = T.Compose([
        T.Resize((size,size)),
        T.RandomAffine(10),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.ToTensor(),
        T.Normalize((0.507395516207, ),(0.255128989415, )) 
    ])
    return tmf
    

def fer2013_valid_transform_fn(size=48):
    tmf = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize((0.507395516207, ),(0.255128989415, ))
    ])  
    return tmf

def fer2013_example_transform_fn(size=48):
    tmf = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize((0.507395516207, ),(0.255128989415, ))
    ])  
    return tmf


train_transform = T.Compose([
                T.RandomAffine(10),
                T.RandomHorizontalFlip(),
                T.RandomRotation(30),
                T.ToTensor(),
                T.Normalize((0.507395516207, ),(0.255128989415, )) 
                ])
val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.507395516207, ),(0.255128989415, ))
            ])  


class IBug300WDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, num_workers=16,
                 image_size=224, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage: Optional[str] = None):
        print(f'Preparing datamodule for stage {stage}, please wait...')
        self.trainset = IBug300WDataset(root=self.data_dir, train=True, transform=train_landmarks_transform_fn(size=self.image_size))
        self.validset = IBug300WDataset(root=self.data_dir, train=False, transform=valid_landmarks_transform_fn(size=self.image_size))
        self.predset = IBug300WDataset(root=self.data_dir, train=False, transform=valid_landmarks_transform_fn(size=self.image_size))
        
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
                 image_size=224, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
    def setup(self, stage: Optional[str] = None):
        print(f'Preparing datamodule for stage {stage}, please wait...')
        self.trainset = IBugDLib300WDataset(root=self.data_dir, train=True, transform=train_landmarks_transform_fn(size=self.image_size))
        self.validset = IBugDLib300WDataset(root=self.data_dir, train=False, transform=valid_landmarks_transform_fn(size=self.image_size))
        self.predset = IBugDLib300WDataset(root=self.data_dir, train=False, transform=valid_landmarks_transform_fn(size=self.image_size))
        self.dataset = IBugDLib300WDataset(root=self.data_dir, train=True, transform=example_landmarks_transform_fn())

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
        
        self.image_size = image_size
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: Optional[str] = None):
        print(f'Preparing datamodule for stage {stage}, please wait...')
        self.trainset = FER2013(root=self.data_dir, mode='train', transform=fer2013_train_transform_fn(size=self.image_size))
        self.validset = FER2013(root=self.data_dir, mode='valid', transform=fer2013_valid_transform_fn(size=self.image_size))
        self.testset = FER2013(root=self.data_dir, mode='test', transform=fer2013_valid_transform_fn(size=self.image_size))
        self.dataset = FER2013(root=self.data_dir, mode='test', transform=fer2013_example_transform_fn(size=self.image_size))
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size,  num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size,  num_workers=self.num_workers)
        
