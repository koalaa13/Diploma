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
        self.conv_trans = nn.ConvTranspose2d(in_channels=1, out_channels=7, kernel_size=(5, 5), stride=(1, 1),
                                             padding=(2, 2), dilation=(1, 1), groups=1)
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=7, out_channels=2, kernel_size=(2, 2), stride=(3, 3),
                                              padding=(1, 1),
                                              dilation=(1, 1), groups=1)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=2, out_channels=22, kernel_size=(3, 3), stride=(1, 1),
                                              padding=(1, 1),
                                              dilation=(1, 1), groups=1)
        self.conv_trans3 = nn.ConvTranspose2d(in_channels=22, out_channels=28, kernel_size=(4, 4), stride=(1, 1),
                                              padding=(1, 1),
                                              dilation=(2, 2), groups=1)
        self.conv_trans4 = nn.ConvTranspose2d(in_channels=28, out_channels=15, kernel_size=(2, 2), stride=(2, 2),
                                              padding=(0, 0),
                                              dilation=(2, 2), groups=1)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        x = self.conv_trans3(x)
        x = self.conv_trans4(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        return x
