import json

import torch
import torch.nn as nn

from embedding.graph import NeuralNetworkGraph

# 0.emb = ALEXNET
with open('./estimator/graph_generated_embeddings/good_embedding.txt') as f:
    embedding = json.load(f)

for j in embedding:
    a = j[:37]
    for id, i in enumerate(a):
        if i is not None:
            print(id, i)
    print('edges count: ' + str(j[37]))
    print('####################\n')

