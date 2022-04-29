import importlib
import os
import sys

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph, ATTRIBUTES_POS_COUNT, NODE_EMBEDDING_DIMENSION, node_to_ops, \
    attribute_to_pos
from tmp import tmp

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

from utils.DatasetTransformer import Transformer
from utils.Mapper import Mapper
from wgan.Generator import Generator

small_mapper = Mapper()
os.makedirs('./data/small_dims_part_mapped', exist_ok=True)
small_mapper.map_to_super_small_embedding('./data/small_dims_parts', './data/small_dims_part_mapped')

big_mapper = Mapper()
os.makedirs('./data/big_dims_part_mapped', exist_ok=True)
big_mapper.map_to_super_small_embedding('./data/big_dims_parts', './data/big_dims_part_mapped')

# m = Mapper()
# m.split_to_blocks('./data/nn_embedding', './data/big_dims_parts', './data/small_dims_parts')
# for file in os.listdir('./data/nn_embedding'):
#     with open(os.path.join('./data/big_dims_parts', file)) as f:
#         big_dim = json.load(f)
#     with open(os.path.join('./data/small_dims_parts', file)) as f:
#         small_dim = json.load(f)
#     big_dim_not_null = 0
#     small_dim_not_null = 0
#     while big_dim_not_null < len(big_dim) and big_dim[big_dim_not_null][19] is not None:
#         big_dim_not_null += 1
#     while small_dim_not_null < len(small_dim) and small_dim[small_dim_not_null][19] is not None:
#         small_dim_not_null += 1
#     print(big_dim_not_null, small_dim_not_null)
#     assert big_dim_not_null + small_dim_not_null == 10

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

big_embedding_height = 5
latent_dim = 5
big_output_generator_dim = big_embedding_height * embedding_width
big_obj_shape = (big_embedding_height, embedding_width)
big_generator_dims = [latent_dim, 10, big_output_generator_dim]

big_generator = Generator(big_generator_dims, big_obj_shape).to(device)
big_generator.load_state_dict(torch.load('./wgan/big_dims_generator_weights'))
big_generator.eval()

small_embedding_height = 9
small_output_generator_dim = small_embedding_height * embedding_width
small_obj_shape = (small_embedding_height, embedding_width)
small_generator_dims = [latent_dim, 10, small_output_generator_dim]

small_generator = Generator(small_generator_dims, small_obj_shape).to(device)
small_generator.load_state_dict(torch.load('./wgan/small_dims_generator_weights'))
small_generator.eval()

print('TRANSFORMATION STARTED')
os.makedirs('./data/big_dims_part_mapped_transformed', exist_ok=True)
big_transformer = Transformer(embedding_width, big_embedding_height)
big_transformer.transform_dataset('./data/big_dims_part_mapped',
                                  './data/big_dims_part_mapped_transformed')
print('TRANSFORMATION FINISHED')
print(big_transformer.mns)
print(big_transformer.mxs)

print('TRANSFORMATION STARTED')
os.makedirs('./data/small_dims_part_mapped_transformed', exist_ok=True)
small_transformer = Transformer(embedding_width, small_embedding_height)
small_transformer.transform_dataset('./data/small_dims_part_mapped',
                                    './data/small_dims_part_mapped_transformed')
print('TRANSFORMATION FINISHED')
print(small_transformer.mns)
print(small_transformer.mxs)

its = 1000
z = torch.randn(its, latent_dim).to(device)
big_parts = big_generator(z).detach().cpu().numpy().tolist()
small_parts = small_generator(z).detach().cpu().numpy().tolist()
for i in range(its):
    big_transformer.de_transform_embedding(big_parts[i])
    small_transformer.de_transform_embedding(small_parts[i])
    whole_network = []
    in_shape = [1, 28, 28]

    big_parts[i], in_shape = big_mapper.de_map_from_super_small_embedding(big_parts[i], in_shape)

    flatten = [None] * NODE_EMBEDDING_DIMENSION
    flatten[attribute_to_pos['op']] = node_to_ops['Flatten']

    assert len(in_shape) == 3
    flatten[5] = 1  # axis
    flatten[20] = 1  # batch_size
    flatten[21] = in_shape[0] * in_shape[1] * in_shape[2]
    print('CHANNELS = ' + str(in_shape[0]))
    in_shape = [1, in_shape[0] * in_shape[1] * in_shape[2]]

    for jj in big_parts[i]:
        whole_network.append(jj)
    whole_network.append(flatten)

    assert len(in_shape) == 2
    small_parts[i], in_shape = small_mapper.de_map_from_super_small_embedding(small_parts[i][0:5], in_shape)
    small_parts[i] = small_parts[i][0:5]
    for jj in small_parts[i]:
        whole_network.append(jj)

    flatten = [None] * NODE_EMBEDDING_DIMENSION
    linear = [None] * NODE_EMBEDDING_DIMENSION
    log_softmax = [None] * NODE_EMBEDDING_DIMENSION

    flatten[attribute_to_pos['op']] = node_to_ops['Flatten']

    print(in_shape)
    flatten[5] = 1
    flatten[20] = 1  # batch_size
    flatten[21] = in_shape[1]

    linear[attribute_to_pos['op']] = node_to_ops['Linear']

    linear[0] = 1.0
    out_channel = 10
    linear[20] = 1
    linear[21] = out_channel

    in_shape = [1, out_channel]

    log_softmax[attribute_to_pos['op']] = node_to_ops['LogSoftmax']

    log_softmax[5] = 1
    log_softmax[20] = in_shape[0]
    log_softmax[21] = in_shape[1]

    whole_network.append(flatten)
    whole_network.append(linear)
    whole_network.append(log_softmax)

    for i in range(len(whole_network)):
        if i != len(whole_network) - 1:
            whole_network[i][ATTRIBUTES_POS_COUNT] = 1
            whole_network[i][ATTRIBUTES_POS_COUNT + 1] = i + 1
        else:
            whole_network[i][ATTRIBUTES_POS_COUNT] = 0

    for id1, j in enumerate(whole_network):
        print(id1)
        a = j
        for id, i in enumerate(a):
            if i is not None:
                print(id, i)
        print('edges count: ' + str(j[ATTRIBUTES_POS_COUNT]))
        print('####################\n')
    graph = NeuralNetworkGraph.get_graph(whole_network)
    os.makedirs('./tmp', exist_ok=True)
    Converter(graph, filepath='./tmp/tmp.py', model_name='Tmp')

    importlib.reload(tmp)
    n_epoch = 10
    model = tmp.Tmp().to(device)
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

    sys.exit(1)

# for i in range(its):
#     print("ITERATION " + str(i) + " STARTED")
#     try:
#         small_embedding = fake[i]
#         print(small_embedding)
#         full_embedding = m.de_map_from_super_small_embedding(small_embedding, [1, 28, 28])
#         print('full embedding generated')
#         graph = NeuralNetworkGraph.get_graph(full_embedding)
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
#             f.write(json.dumps(full_embedding))
#     except Exception as e:
#         print(str(e))
#     print("ITERATION " + str(i) + " FINISHED")
