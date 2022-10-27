import torch
import torch.nn as nn
import numpy as np
# from . import common
import torchvision.models as models
from torchvision.models import ViT_L_16_Weights


class VitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.classifier = nn.Sequential(
            nn.Dropout(True),
            # nn.Linear(1000, 128),
            nn.Linear(1000, 2),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        # x = self.autoencoder(x)
        x = self.module.forward(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

# class

if __name__ == '__main__':
    model = VitNet()
    print(model)
    input = torch.randn(4, 3, 512, 512)
    out = model(input)
    print(out)   