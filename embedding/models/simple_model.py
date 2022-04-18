import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(784, 10)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(1, 20, (3, 3))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = torch.flatten(x, 1)
        return x
