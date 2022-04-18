import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=1):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=20, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1),
            nn.Flatten(),
            nn.Flatten(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
