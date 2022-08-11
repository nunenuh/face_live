from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class NaimishNet(nn.Module):
    def __init__(self, num_pts=136):
        super(NaimishNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2,2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=1, padding=0)
        
        self.dropout_conv1 = nn.Dropout(p=0.1)
        self.dropout_conv2 = nn.Dropout(p=0.2)
        self.dropout_conv3 = nn.Dropout(p=0.3)
        self.dropout_conv4 = nn.Dropout(p=0.4)
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        
        self.fc1 = nn.Linear(in_features=6400, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=num_pts)
        
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.dropout_fc2 = nn.Dropout(p=0.6)
        
        # self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.uniform(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform(m.weight)
        
    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout_conv1(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout_conv2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout_conv3(x)
    
        x = self.pool(F.elu(self.conv4(x)))
        x = self.dropout_conv4(x)
        
        # print(x.shape)
        
        x = x.view(x.size(0), -1)
        
        x = F.elu(self.fc1(x))
        x = self.dropout_fc1(x)
        
        x = F.elu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x        
        

class FaceLandmarkNet(nn.Module):
    def __init__(self, backbone_name:str = "efficient_net", num_pts:int=68):
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
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    
    def _get_backbone(self, backbone:str = "efficient_net"):
        if backbone== "efficient_net":
            return efficientnet_v2_s(num_classes=1024)
        elif backbone== "mobilenet_v3":
            return mobilenet_v3(num_classes=1024)
        elif backbone== "quantized_mobilenet_v3":
            return quantized_mobilenet_v3(num_classes=1024)
        else:
            raise Exception("Backbone not supported")
    
def efficientnet_v2_s(weights_path=None, num_classes=512):
    from torchvision.models import efficientnet, EfficientNet_V2_S_Weights
    env2s = efficientnet.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    env2s.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    env2s.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    if weights_path != None:
        if Path(weights_path).exists():
            env2s.load_state_dict(torch.load(weights_path))
    
    return env2s

def mobilenet_v3(weights_path=None, num_classes=512):
    from torchvision.models import mobilenetv3, MobileNet_V3_Small_Weights
    mnv3 = mobilenetv3.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    mnv3.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    mnv3.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            mnv3.load_state_dict(torch.load(weights_path))
    
    return mnv3


def quantized_mobilenet_v3(weights_path=None, num_classes=512):
    from torchvision.models import quantization
    qmvn3 = quantization.mobilenet_v3_large(weights=quantization.MobileNet_V3_Large_QuantizedWeights)
    qmvn3.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    qmvn3.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    
    if weights_path != None:
        if Path(weights_path).exists():
            qmvn3.load_state_dict(torch.load(weights_path))
        
    return qmvn3



if __name__ == "__main__":
    net = NaimishNet(num_pts=136)
    input = torch.rand(32,1,96,96)
    net(input)