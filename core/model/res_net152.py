import torch
import torch.nn as nn
import numpy as np
# from . import common
import torchvision.models as models


class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = models.resnet18(pretrained=True)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.Linear(1000, 128),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x = self.autoencoder(x)
        x = self.module.forward(x)
        x = self.classifier(x)
        # x = nn.functional.softmax(x)
        return x


if __name__ == '__main__':
    model = ResNet152()
    print(model)
    input = torch.randn(4, 3, 224, 224)
    out = model(input)
    print(out)   