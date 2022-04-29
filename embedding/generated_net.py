import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=1):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_shape, out_channels=7, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1),
            nn.ConvTranspose2d(in_channels=7, out_channels=2, kernel_size=(2, 2), stride=(3, 3), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ConvTranspose2d(in_channels=2, out_channels=22, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1),
            nn.ConvTranspose2d(in_channels=22, out_channels=28, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), dilation=(2, 2), groups=1),
            nn.ConvTranspose2d(in_channels=28, out_channels=15, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(2, 2), groups=1),
            nn.Flatten(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
