import torch.nn as nn
import torch.nn.functional as F
import torch
from . import util


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.Dropout2d(0.3),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
        )
        

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.3),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.3),
        )

        self.final = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 1))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            util.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(F.leaky_relu(x, 0.2))
        x = self.layer3(F.leaky_relu(x, 0.2))
        x = self.layer4(F.leaky_relu(x, 0.2))
        x = self.final(F.leaky_relu(x, 0.2))
        x = torch.sigmoid(x)

        return x
