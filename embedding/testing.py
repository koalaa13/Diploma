import json

import torch

from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph
from embedding.models.simple_model import Net

model = Net()
xs = torch.zeros(1, 1, 28, 28)
g = NeuralNetworkGraph(model=model, test_batch=xs)
embedding = g.get_naive_embedding()
widths = set()
print(len(embedding))
for i in range(len(embedding)):
    widths.add(len(embedding[i]))
print(widths)
with open('./embedding', 'w+') as f:
    f.write(json.dumps(embedding))

graph = NeuralNetworkGraph.get_graph(embedding)
Converter(graph, filepath='./generated_net.py', model_name='Tmp')
