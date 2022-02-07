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

        modules = [*block(dims[0], dims[1], normalize=False)]
        for i in range(1, len(dims) - 2):
            modules.extend(block(dims[i], dims[i + 1]))
        modules.append(nn.Linear(dims[len(dims) - 2], dims[len(dims) - 1]))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, z, obj_shape):
        # mb resize result here
        img = self.model(z)
        img = img.view(img.shape[0], *obj_shape)
        return self.model(z)
