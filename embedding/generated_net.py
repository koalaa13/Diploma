import torch
from torch import nn


class Tmp(nn.Module):

    def __init__(self, in_shape=28 * 28):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=10),
            nn.ReLU(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
