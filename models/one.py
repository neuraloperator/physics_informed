import torch
import torch.nn as nn
from .basics import SpectralConv3d


class FNet(nn.Module):
    def __init__(self, mode1, mode2, mode3):
        super(FNet, self).__init__()
        self.mode1 = mode1
        self.mode2 = mode2
        self.mode3 = mode3
        self.sp = SpectralConv3d(1, 1, mode1, mode2, mode3)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = x[:, -1:, :, :, :]
        y = self.sp(x)
        return y