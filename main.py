import json
import math
import sys

import torch
import torch.nn as nn
import torchvision
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph, ATTRIBUTES_POS_COUNT

# 0.emb = ALEXNET
with open('./estimator/estimator_generated_embeddings/0_9.8.txt') as f:
    embedding = json.load(f)

for j in embedding:
    a = j[:ATTRIBUTES_POS_COUNT]
    for id, i in enumerate(a):
        if i is not None:
            print(id, i)
    print('edges count: ' + str(j[ATTRIBUTES_POS_COUNT]))
    print('####################\n')

graph = NeuralNetworkGraph.get_graph(embedding)
Converter(graph, filepath='./generated_net.py', model_name='Tmp')

# from generated_net import Tmp
#
# train_dataloader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data/mnist',
#                                train=True,
#                                download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=64,
#     shuffle=True)
#
# test_dataloader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST("./data/mnist",
#                                train=False,
#                                download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=1000,
#     shuffle=True)
#
# cuda = torch.cuda.is_available()
# device = torch.device('cuda:0') if cuda else torch.device('cpu')
# n_epoch = 5
# model = Tmp().to(device)
# for name, param in model.named_parameters():
#     print(name)
#     print(param)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# for epoch in range(n_epoch):
#     for i, (data, target) in enumerate(tqdm(train_dataloader)):
#         data = data.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(data).to(device)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
# print('MODEL TRAINING FINISHED')
# # testing and calculating accuracy
# for name, param in model.named_parameters():
#     print(name)
#     print(param)
# model.eval()
# correct = 0
# with torch.no_grad():
#     for i, (data, target) in enumerate(tqdm(test_dataloader)):
#         data = data.to(device)
#         # print(data.data)
#         target = target.to(device)
#         output = model(data).to(device)
#         # print('TARGET: ' + str(target.data))
#         # print('OUTPUT: ' + str(output.data))
#         pred = output.data.max(1, keepdim=True)[1]
#         # print('PRED: ' + str(pred.data))
#         correct += pred.eq(target.data.view_as(pred)).sum()
#         # if i == 10:
#         #     sys.exit(0)
# accuracy = 100. * correct.item() / len(test_dataloader.dataset)
# print(accuracy)
