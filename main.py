import copy
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
import hiddenlayer as hl

# with open('./generated/35.txt') as f:
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

from generated_net import Tmp
from utils.DatasetTransformer import Transformer
from utils.Mapper import Mapper
from wgan.Discriminator import Discriminator
from wgan.Generator import Generator

m = Mapper()
os.makedirs('./data/nn_super_small_embedding', exist_ok=True)
m.map_to_super_small_embedding('./data/nn_embedding', './data/nn_super_small_embedding')

train_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/mnist',
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
    torchvision.datasets.MNIST("./data/mnist",
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

embedding_width = 1
embedding_height = 14
latent_dim = 5
output_generator_dim = embedding_height * embedding_width
obj_shape = (embedding_height, embedding_width)
generator_dims = [latent_dim, 10, output_generator_dim]
generator = Generator(generator_dims, obj_shape).to(device)
generator.load_state_dict(torch.load('./wgan/generator_weights'))
generator.eval()

# z = torch.randn(1, latent_dim).to(device)
# hl_graph = hl.build_graph(generator, z, transforms=None)
# hl_graph.theme = hl.graph.THEMES['blue'].copy()
# hl_graph.save('GAN Generator', format='png')

# discriminator_dims = [output_generator_dim, 512, 256]
# discriminator = Discriminator(discriminator_dims).to(device)
# z = torch.randn(1, embedding_height, embedding_width).to(device)
# hl_graph = hl.build_graph(discriminator, z, transforms=None)
# hl_graph.theme = hl.graph.THEMES['blue'].copy()
# hl_graph.save('GAN Discriminator', format='png')


print('TRANSFORMATION STARTED')
transformer = Transformer(embedding_width, embedding_height)
transformer.transform_dataset('./data/nn_super_small_embedding',
                              './data/nn_super_small_embedding_transformed')
print('TRANSFORMATION FINISHED')
print(transformer.mns)
print(transformer.mxs)
#
# eps = 1e-8
#
# for file in os.listdir('./data/nn_embedding'):
#     with open(os.path.join('./data/nn_embedding/', file)) as f:
#         original_embedding = json.load(f)
#     with open(os.path.join('./data/nn_embedding_transformed/', file)) as f:
#         transformed_embedding = json.load(f)
#     with open(os.path.join('./data/nn_embedding_transformed/', file)) as f:
#         norm_embedding = json.load(f)
#     transformer.de_transform_embedding(transformed_embedding)
#     assert len(original_embedding) == embedding_height
#     assert len(transformed_embedding) == embedding_height
#     assert len(norm_embedding) == embedding_height
#     if len(transformed_embedding) != len(original_embedding):
#         print('Lens are not equal ' + str(file))
#         sys.exit(1)
#     for j in range(len(transformed_embedding)):
#         if len(transformed_embedding[j]) != len(original_embedding[j]):
#             print('Lens ' + str(j) + ' are not equal ' + str(file))
#             sys.exit(2)
#         for k in range(len(transformed_embedding[j])):
#             if transformed_embedding[j][k] != original_embedding[j][k]:
#                 if transformed_embedding[j][k] is None or original_embedding[j][k] is None:
#                     print(file, j, k)
#                     print(original_embedding[j][19])
#                     print(norm_embedding[j][k])
#                     print(transformed_embedding[j][k])
#                     print(original_embedding[j][k])
#                     print(transformer.mns[k])
#                     print(transformer.mxs[k])
#                     sys.exit(3)
#                 if math.fabs(transformed_embedding[j][k] - original_embedding[j][k]) > eps:
#                     print(file, j, k)
#                     print(norm_embedding[j][k])
#                     print(transformed_embedding[j][k])
#                     print(original_embedding[j][k])
#                     print(transformer.mns[k])
#                     print(transformer.mxs[k])
#                     sys.exit(4)

print('GENERATING EMBEDDINGS')
os.makedirs('./generated', exist_ok=True)
its = 1000
z = torch.randn(its, latent_dim).to(device)
fake = generator(z).detach().cpu().numpy().tolist()
for i in range(its):
    transformer.de_transform_embedding(fake[i])
    with open('./generated/' + str(i) + '.txt', 'w+') as f:
        f.write(json.dumps(fake[i]))
print('GENERATING EMBEDDINGS FINISHED')

for i in range(its):
    print("ITERATION " + str(i) + " STARTED")
    try:
        small_embedding = fake[i]
        print(small_embedding)
        full_embedding = m.de_map_from_super_small_embedding(small_embedding, [1, 28, 28])
        print('full embedding generated')
        graph = NeuralNetworkGraph.get_graph(full_embedding)
        Converter(graph, filepath='./generated_net.py', model_name='Tmp')

        n_epoch = 10
        model = Tmp().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(n_epoch):
            for j, (data, target) in enumerate(tqdm(train_dataloader)):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data).to(device)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        print('MODEL TRAINING FINISHED')
        # testing and calculating accuracy
        for name, param in model.named_parameters():
            print(name)
            print(param)
        model.eval()
        correct = 0
        with torch.no_grad():
            for j, (data, target) in enumerate(tqdm(test_dataloader)):
                data = data.to(device)
                # print(data.data)
                target = target.to(device)
                output = model(data).to(device)
                # print('TARGET: ' + str(target.data))
                # print('OUTPUT: ' + str(output.data))
                pred = output.data.max(1, keepdim=True)[1]
                # print('PRED: ' + str(pred.data))
                correct += pred.eq(target.data.view_as(pred)).sum()
                # if i == 10:
                #     sys.exit(0)
        accuracy = 100. * correct.item() / len(test_dataloader.dataset)
        print(accuracy)
        os.makedirs('./generated_successfully', exist_ok=True)
        with open('./generated_successfully/' + str(i) + '_' + str(accuracy) + '.txt', 'w+') as f:
            f.write(json.dumps(full_embedding))
    except Exception as e:
        print(str(e))
    print("ITERATION " + str(i) + " FINISHED")
