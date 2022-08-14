import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image

from pathlib import Path
import numpy as np
import pandas as pd
import xmltodict

class IBug300WDataset(Dataset):
    def __init__(self, root, train=True, val_size=0.2, transform=None):
        self.root = root
        self.train = train
        self.val_size = val_size
        self.transform = transform
        self.dframe = self._dframe()
    
    def _load_image(self, path, to_grayscale=False):
        img = Image.open(path)
        if to_grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        return np.array(img).astype(np.uint8)
    
    def _load_text_file(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        return data
    
    def _load_points(self, path):
        data = self._load_text_file(path)
        
        start_idx, end_idx = 0, len(data)-1
        for idx, line in enumerate(data):
            if line.startswith('{'):
                start_idx = idx+1
                break;
        
        string_pts = data[start_idx:end_idx]
        points = []
        for pts in string_pts:
            pts = pts.replace("\n","")
            sx,sy = str_xy = pts.split(" ")
            x, y = float(sx.strip()), float(sy.strip())
            points.append((x,y))
        
        return np.array(points)
    
    def _get_files(self, dir_prefix, pattern="*.png"):
        path = Path(self.root).joinpath(dir_prefix)
        files = list(path.glob(pattern))
        files.sort()
        return files
    
    def _create_dframe(self, dir_prefix):
        pts_files = self._get_files(dir_prefix=dir_prefix, pattern="*.pts")
        img_files = self._get_files(dir_prefix=dir_prefix, pattern="*.png")

        data = {'images':[], 'points':[]}
        for img_path, pts_path in zip(img_files, pts_files):
            if img_path.stem == pts_path.stem:
                data['images'].append(img_path)
                data['points'].append(pts_path)
        
        return pd.DataFrame(data)
    
    def _dframe(self):
        indoor_frame = self._create_dframe("01_Indoor")
        outdoor_frame = self._create_dframe("02_Outdoor")
        
        from sklearn.model_selection import train_test_split
        indoor_train, indoor_valid = train_test_split(indoor_frame, test_size=self.val_size)
        outdoor_train, outdoor_valid = train_test_split(outdoor_frame, test_size=self.val_size)
        
        self.train_frame = pd.concat([indoor_train, outdoor_train])
        self.valid_frame = pd.concat([indoor_valid, outdoor_valid])
        
        if self.train:
            return self.train_frame
        else:
            return self.valid_frame
        
    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.dframe.iloc[idx]
        
        image = self._load_image(str(img_path))
        points = self._load_points(str(pts_path))
        
        if self.transform:
            image, points = self.transform(image, points)
        
        return {'image': image, 'landmark': points}
        
class IBugDLib300WDataset(Dataset):
    def __init__(self, root, train=True, transform=None, 
            train_xml_file = 'labels_ibug_300W_train.xml',
            test_xml_file = 'labels_ibug_300W_test.xml',
        ):
        self.root = root
        self.train = train
        self.xml_train_file = Path(root).joinpath(train_xml_file)
        self.xml_valid_file = Path(root).joinpath(test_xml_file)
        self.transform = transform
        # self.dframe = self._dframe()
        self.xmldata = self._xmldata()
        
    def _load_image(self, path, to_grayscale=False):
        img = Image.open(path)
        if to_grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
            
        return np.array(img).astype(np.uint8)
    
    def _load_xml_to_dict(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            xmldata = file.read()
        xml2dict = xmltodict.parse(xmldata)
        
        return xml2dict
    
    def _xmldata(self):
        fpath = self.xml_train_file
        if not self.train:
            fpath = self.xml_valid_file
        
        gdata = self._load_xml_to_dict(str(fpath))
        xmldata = gdata['dataset']['images']['image']
        return xmldata
    
    def _image_path(self, data):
        return Path(self.root).joinpath(data["@file"])
    
    def _get_size(self, data):
        w, h = data["@width"], data["@height"]
        return w, h
    
    def _get_box(self, data):
        left = float(data["box"]["@left"])
        top = float(data["box"]["@top"])
        width = float(data["box"]["@width"])
        height = float(data["box"]["@height"])

        return left, top, width, height
    
    def _get_points(self, data):
        parts = data["box"]["part"]
        points = []
        for p in parts:
            n,x,y = p["@name"], float(p["@x"]), float(p["@y"])
            points.append((x,y))
            
        return np.array(points)
    
    def __len__(self):
        return len(self.xmldata)
    
    def __getitem__(self, idx):
        data = self.xmldata[idx]
        # print(data)
        
        impath = self._image_path(data)
        impath = str(Path(self.root).joinpath(impath))
        
        image = self._load_image(impath)
        points = self._get_points(data)
        
        
        if self.transform:
            image, points = self.transform(image, points)
        
        return {'image': image, 'landmark': points}
    
    def _scaled_by_facebox(self, data, image, points, scale = 1.5):
        scaler = (scale-1)/2
        x, y, w, h = self._get_box(data)
        xmin,ymin,xmax,ymax = x,y,x+w,y+h
        ymin = ymin - int(scaler * h)
        ymax = ymax + int(scaler * h)
        xmin = xmin - int(scaler * w)
        xmax = xmax + int(scaler * w)
        
        if ymin<=0: ymin = 0
        if xmin<=0: xmin = 0
        if xmax>=image.shape[1]: xmax = image.shape[1]
        if ymax>=image.shape[0]: ymax = image.shape[0]
        
        # points operation
        points = [[(xp-xmin), (yp-ymin)] for xp,yp in points.tolist()]
        points = np.array(points)
        # points = points - [xmin/w, ymin/h]
        # points = points / [w, h]
        
        #image crop
        face = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        return face, points

        
        
    def getitem(self, idx, scale=1.5):
        data = self.xmldata[idx]
        # print(data)
        
        impath = self._image_path(data)
        impath = str(Path(self.root).joinpath(impath))
        
        image = self._load_image(impath)
        print(self._get_box(data))
        points = self._get_points(data)
        
        image, points = self._scaled_by_facebox(data, image, points, scale=scale)
        
        
        if self.transform:
            image, points = self.transform(image, points)
        
        return {'image': image, 'landmark': points}
        
    

