import torch
import torch.nn as nn
import numpy as np
# from . import common
import torchvision.models as models

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = models.resnet18(pretrained = True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 2), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.module.forward(x)
        x = self.classifier(x)
        return x

# class

if __name__ == '__main__':
    model = BaselineNet()
    print(model)
    
    input = torch.randn(4, 3, 224, 224)
    out = model(input)
    print(out)   