from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class NaimishNet(nn.Module):
    def __init__(self, num_pts=136):
        super(NaimishNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4,4)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Dropout(p=0.1),
         
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Dropout(p=0.2),
         
            nn.Conv2d(64, 128, kernel_size=(2,2), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Dropout(p=0.3),
            
            nn.Conv2d(128, 256, kernel_size=(1,1), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Dropout(p=0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=6400, out_features=1000),
            nn.ELU(),
            nn.Dropout(p=0.5),
                       
            nn.Linear(in_features=1000, out_features=500),
            nn.ELU(),
            nn.Dropout(p=0.6),
            
            nn.Linear(in_features=500, out_features=num_pts),
        )
        
        # self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.uniform(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform(m.weight)
                
    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True
        
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def freeze(self):
        self.freeze_backbone()
        self.freeze_classifier()
    
    def unfreeze(self):
        self.unfreeze_backbone()
        self.unfreeze_classifier()
    
    
    def forward(self, x):
        x = self.features(x)
        
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        
        return x        
        

class FaceLandmarkNet(nn.Module):
    def __init__(self, backbone_name:str = "efficientnet_v2s", num_pts:int=68):
        super(FaceLandmarkNet, self).__init__()
        self.num_pts = num_pts
        self.backbone: nn.Module = self._get_backbone(backbone_name)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            
            nn.Dropout(p=0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_pts),
        )
        
    def _get_backbone(self, name):
        if name=="efficientnet_v2s":
            return efficientnet_v2s(num_classes=1024)
        elif name=="efficientnet_v2m":
            return efficientnet_v2m(num_classes=1024)
        elif name=="efficientnet_v2l":
            return efficientnet_v2l(num_classes=1024)
        elif name=="mobilenet_v3":
            return mobilenet_v3(num_classes=1024)
        elif name=="quantized_mobilenet_v3":
            return quantized_mobilenet_v3(num_classes=1024)
        else:
            raise Exception("Unknown backbone")
        
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def freeze(self):
        self.freeze_backbone()
        self.freeze_classifier()
    
    def unfreeze(self):
        self.unfreeze_backbone()
        self.unfreeze_classifier()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

def efficientnet_v2l(weights_path=None, num_classes=1024):
    from torchvision.models import efficientnet,  EfficientNet_V2_L_Weights
    
    env2s = efficientnet.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    # env2s.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    env2s.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            env2s.load_state_dict(torch.load(weights_path))
    
    return env2s
   

def efficientnet_v2m(weights_path=None, num_classes=1024):
    from torchvision.models import efficientnet,  EfficientNet_V2_M_Weights
    
    env2s = efficientnet.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    # env2s.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    env2s.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            env2s.load_state_dict(torch.load(weights_path))
    
    return env2s
   
def efficientnet_v2s(weights_path=None, num_classes=1024):
    from torchvision.models import efficientnet, EfficientNet_V2_S_Weights
    env2s = efficientnet.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # env2s.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    env2s.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            env2s.load_state_dict(torch.load(weights_path))
    
    return env2s

def mobilenet_v3(weights_path=None, num_classes=512):
    from torchvision.models import mobilenetv3, MobileNet_V3_Small_Weights
    mnv3 = mobilenetv3.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # mnv3.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    mnv3.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            mnv3.load_state_dict(torch.load(weights_path))
    
    return mnv3

def quantized_mobilenet_v3(weights_path=None, num_classes=512):
    from torchvision.models import quantization
    qmvn3 = quantization.mobilenet_v3_large(weights=quantization.MobileNet_V3_Large_QuantizedWeights)
    # qmvn3.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    qmvn3.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            qmvn3.load_state_dict(torch.load(weights_path))
        
    return qmvn3


if __name__ == "__main__":
    net = NaimishNet(num_pts=136)
    input = torch.rand(32,1,96,96)
    net(input)