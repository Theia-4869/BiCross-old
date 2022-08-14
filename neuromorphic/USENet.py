import torch
from torch import nn
import torch.nn.functional as F

from neuromorphic.UNet import UNet
from neuromorphic.SENet import SELayer


class USENet(UNet):
    def __init__(self, n_channels=128, bilinear=True):
        super(USENet, self).__init__(n_channels, bilinear)
        self.se_layer = SELayer(channel=512)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.se_layer(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
