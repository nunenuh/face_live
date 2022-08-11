import torch
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path


import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


class FER2013(Dataset):
    def __init__(self, root, mode='train', val_size=0.2, transform=None):
        super(FER2013, self).__init__()
        self.root = root
        self.mode = mode
        self.val_size = val_size
        self.transform = transform
        
        self.data = self._load_data()
        
    
    def _load_images_fer2013(self):
        if self.mode=="train" or self.mode=="valid":
            csv_file = str(Path(self.root).joinpath('train.csv'))
        elif self.mode =="test":
            csv_file = str(Path(self.root).joinpath('test.csv'))
        else:
            raise Exception("")
        
        dframe = pd.read_csv(csv_file)
        pixels = dframe['pixels'].tolist()
        w, h = 48, 48
        faces = []
        # print(pixels)
        for pseq in pixels:
            # print(pseq)
            face = [int(pix) for pix in pseq.split(' ')]
            face = np.asarray(face).reshape(w,h).astype(np.uint8)
            face = cv2.resize(face, (w,h))
            faces.append(face)
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        return faces
    
    def _load_emotions_fer2013(self):
        if self.mode=="train" or self.mode=="valid":
            csv_file = str(Path(self.root).joinpath('train.csv'))
            dframe = pd.read_csv(csv_file)
            emotions = dframe["emotion"].values
            return emotions
        else:
            return None
        
    def _split_val(self, faces, emotions):
        trn_face, val_face, trn_emo, val_emo = train_test_split(
            faces, emotions, test_size=self.val_size,
            random_state =1261, shuffle=True
        )
        
        return {"faces":trn_face, "emotions":trn_emo}, {"faces":val_face, "emotions":val_emo}
    
    def _load_data(self):
        faces = self._load_images_fer2013()
        emotions = self._load_emotions_fer2013()
        if self.mode=="train" or self.mode=="valid":
            train, valid = self._split_val(faces, emotions)
            if self.mode=="train":
                return train
            else:
                return valid
        elif self.mode=="test":
            return {"faces": faces, "emotions":None}
        else:
            return Exception("")
        
    def _load_to_pil(self, image:np.ndarray):
        image = image.squeeze()
        img = Image.fromarray(image)
        img = img.convert("L")
        return img
    
    def class_to_idx(self, clazz):
        data = {'angry':0, 'disgust':1, 'fear':2, 
                'happy':3, 'sad':4,  'surprise':5, 
                'neutral':6}
        return data.get(clazz, None)
    
    def idx_to_class(self,idx):
        data = ["angry","disgust","fear","happy",
                "sad","surprise","neutral"]
        return data[idx]
        
    def __len__(self):
        return len(self.data["faces"])
        
        
    def __getitem__(self, idx):
        face = self._load_to_pil(self.data["faces"][idx])
        if self.mode!="test":
            emo = self.data["emotions"][idx]
        else:
            emo = None
            
        if self.transform:
            face = self.transform(face)
        
        return face, emo