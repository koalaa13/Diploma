import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(28, 100)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(1, 20, (2, 2))
        self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.logsoftmax(x)
        return x
