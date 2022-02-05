import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dims):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        modules = []
        for i in range(len(dims) - 2):
            modules.append(*block(dims[i], dims[i + 1]))
        modules.append(nn.Linear(dims[len(dims) - 2], dims[len(dims) - 1]))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        # mb resize result here
        return self.model(z)
