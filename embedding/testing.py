import json

import torch

from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph
from embedding.models.simple_model import Net

model = Net()
xs = torch.zeros(1, 1, 28, 28)
# print(model(xs).shape)
g = NeuralNetworkGraph(model=model, test_batch=xs)
embedding = g.get_naive_embedding()

for j in embedding:
    a = j[:37]
    for id, i in enumerate(a):
        if i is not None:
            print(id, i)
    print('####################\n')
# widths = set()
# print(len(embedding))
# for i in range(len(embedding)):
#     widths.add(len(embedding[i]))
# print(widths)
# with open('./embedding', 'w+') as f:
#     f.write(json.dumps(embedding))
#
graph = NeuralNetworkGraph.get_graph(embedding)
Converter(graph, filepath='./generated_net.py', model_name='Tmp')
