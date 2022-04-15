from torch import nn


class Tmp(nn.Module):

    def __init__(self):
        super(Tmp, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(in_features=3, out_features=128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU(),
        )

    def forward(self, x_0):
        x_1 = self.seq1(x_0)
        return x_1
