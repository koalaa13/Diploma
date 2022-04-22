import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph, ATTRIBUTES_POS_COUNT, NODE_EMBEDDING_DIMENSION
from utils.DatasetTransformer import Transformer

# embedding_width = NODE_EMBEDDING_DIMENSION
# embedding_height = 14
# transformer = Transformer(embedding_width, embedding_height)
# transformer.transform_dataset('./data/nn_embedding',
#                               './data/nn_embedding_transformed')
# for i in range(57):
#     with open('./data/nn_embedding_transformed/' + str(i) + '.emb') as f:
#         embedding = json.load(f)
#     transformer.de_transform_embedding(embedding)
#     for j in range(len(embedding) - 1):
#         if embedding[j][ATTRIBUTES_POS_COUNT] != 1:
#             print(i, j)
#             print('edges count is not one')
#             sys.exit(1)
#         ok = embedding[j][ATTRIBUTES_POS_COUNT + 1] == j + 1
#         for k in range(ATTRIBUTES_POS_COUNT + 2, len(embedding[j])):
#             ok &= embedding[j][k] is None
#         if not ok:
#             print(i, j)
#             print('incorrect edges')
#             sys.exit(1)
#     j = len(embedding) - 1
#     if embedding[j][ATTRIBUTES_POS_COUNT] != 0:
#         print(i, j)
#         print('edges count is not one')
#         sys.exit(1)
#     ok = True
#     for k in range(ATTRIBUTES_POS_COUNT + 1, len(embedding[j])):
#         ok &= embedding[j][k] is None
#     if not ok:
#         print(i, j)
#         print('incorrect edges')
#         sys.exit(1)
# print('dataset is ok')

# 0.emb = ALEXNET
# with open('./generated/0.txt') as f:
#     embedding = json.load(f)
#
# for id1, j in enumerate(embedding):
#     print(id1)
#     a = j
#     for id, i in enumerate(a):
#         if i is not None:
#             print(id, i)
#     print('edges count: ' + str(j[ATTRIBUTES_POS_COUNT]))
#     print('####################\n')

# from generated_net import Tmp
# from utils.DatasetTransformer import Transformer
# from wgan.Generator import Generator
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
#
# embedding_width = NODE_EMBEDDING_DIMENSION
# embedding_height = 14
# latent_dim = 100
# output_generator_dim = embedding_height * embedding_width
# obj_shape = (embedding_height, embedding_width)
# generator_dims = [latent_dim, 128, 256, 512, 1024, output_generator_dim]
# generator = Generator(generator_dims).to(device)
# generator.load_state_dict(torch.load('./wgan/generator_weights'))
# generator.eval()
#
# print('TRANSFORMATION STARTED')
# transformer = Transformer(embedding_width, embedding_height)
# transformer.transform_dataset('./data/nn_embedding',
#
#                               './data/nn_embedding_transformed')
# print('TRANSFORMATION FINISHED')
# print(transformer.mns)
# print(transformer.mxs)
#
# print('GENERATING EMBEDDINGS')
# os.makedirs('./generated', exist_ok=True)
# its = 100000
# z = torch.randn(its, latent_dim).to(device)
# fake = generator(z, obj_shape).detach().cpu().numpy().tolist()
# for i in range(its):
#     transformer.de_transform_embedding(fake[i])
#     with open('./generated/' + str(i) + '.txt', 'w+') as f:
#         f.write(json.dumps(fake[i]))
# print('GENERATING EMBEDDINGS FINISHED')
#
# for i in range(its):
#     print("ITERATION " + str(i) + " STARTED")
#     try:
#         graph = NeuralNetworkGraph.get_graph(fake[i])
#         Converter(graph, filepath='./generated_net.py', model_name='Tmp')
#
#         n_epoch = 10
#         model = Tmp().to(device)
#         optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#         for epoch in range(n_epoch):
#             for j, (data, target) in enumerate(tqdm(train_dataloader)):
#                 data = data.to(device)
#                 target = target.to(device)
#                 optimizer.zero_grad()
#                 output = model(data).to(device)
#                 loss = F.nll_loss(output, target)
#                 loss.backward()
#                 optimizer.step()
#         print('MODEL TRAINING FINISHED')
#         # testing and calculating accuracy
#         for name, param in model.named_parameters():
#             print(name)
#             print(param)
#         model.eval()
#         correct = 0
#         with torch.no_grad():
#             for j, (data, target) in enumerate(tqdm(test_dataloader)):
#                 data = data.to(device)
#                 # print(data.data)
#                 target = target.to(device)
#                 output = model(data).to(device)
#                 # print('TARGET: ' + str(target.data))
#                 # print('OUTPUT: ' + str(output.data))
#                 pred = output.data.max(1, keepdim=True)[1]
#                 # print('PRED: ' + str(pred.data))
#                 correct += pred.eq(target.data.view_as(pred)).sum()
#                 # if i == 10:
#                 #     sys.exit(0)
#         accuracy = 100. * correct.item() / len(test_dataloader.dataset)
#         print(accuracy)
#         os.makedirs('./generated_successfully', exist_ok=True)
#         with open('./generated_successfully/' + str(i) + '_' + str(accuracy) + '.txt', 'w+') as f:
#             f.write(json.dumps(fake[i]))
#     except Exception as e:
#         print(str(e))
#     print("ITERATION " + str(i) + " FINISHED")
