import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T

class Vectoriser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        # preprocess = weights.transforms()
        res18 = resnet18(weights=weights)

        modules = list(res18.children())
        # remove max pooling layer and the output layer
        modules = modules[0:3] + modules[4:-1]
        res18 = torch.nn.Sequential(*modules)
        self.res18 = res18

        self.preprocess = T.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
           )

    def forward(self, x):
        x = x / 255.
        x = self.preprocess(x)
        x = self.res18(x)
        x = x.view(x.size(0), -1)
        return x
