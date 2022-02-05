import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dims):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        modules = []
        for i in range(len(dims) - 1):
            modules.extend(block(dims[i], dims[i + 1]))
        modules.append(nn.Linear(dims[len(dims) - 1], 1))
        self.model = nn.Sequential(*modules)

    def forward(self, obj):
        # mb resize input obj here
        return self.model(obj)
