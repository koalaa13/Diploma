import os

import torch
import torchvision

from embedding.graph import NODE_EMBEDDING_DIMENSION, ATTRIBUTES_POS_COUNT
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
    batch_size=64,
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
    batch_size=1000,
    shuffle=True)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if cuda else torch.device('cpu')
estimator = Estimator(NODE_EMBEDDING_DIMENSION, 10, train_dataloader, test_dataloader, device)
os.makedirs('./saved_estimator1', exist_ok=True)
estimator.save('./saved_estimator1')
# for i in range(len(estimator.bad_center)):
#     print(i, estimator.good_center[i][ATTRIBUTES_POS_COUNT], estimator.good_center[i][ATTRIBUTES_POS_COUNT + 1])
# print(len(estimator.bad_center))
# print(len(estimator.good_center))
# print(estimator.good_center)
# print(estimator.bad_center)
