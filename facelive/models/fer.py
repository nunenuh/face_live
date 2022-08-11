from pathlib import Path
import torch
import torch.nn as nn


class FERNet(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_v2s", num_classes:int=7):
        super(FERNet, self).__init__()
        self.num_classes = num_classes
        self.backbone: nn.Module  = self._get_backbone(backbone_name)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(in_features=512, out_features=512),
            nn.Dropout(p=0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(in_features=512, out_features=num_classes),
        )
    
    def _get_backbone(self, name):
        if name=="efficientnet_v2s":
            return efficientnet_v2_s(num_classes=512)
        elif name=="mobilenet_v3":
            return mobilenet_v3(num_classes=512)
        elif name=="quantized_mobilenet_v3":
            return quantized_mobilenet_v3(num_classes=512)
        else:
            raise Exception("Unknown backbone")
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    
    
def efficientnet_v2_s(weights_path=None, num_classes=512):
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

