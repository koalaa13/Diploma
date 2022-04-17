import torch
import torchvision

from estimator.Estimator import Estimator
import torch.nn.functional as F

train_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/mnist',
                               train=True,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=10,
    shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("../data/mnist",
                               train=False,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=10,
    shuffle=True)

estimator = Estimator(100, 10, train_dataloader, test_dataloader, F.nll_loss)
# print(estimator.good_center)
# print(estimator.bad_center)
