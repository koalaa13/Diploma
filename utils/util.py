import json
import os
import shutil
import sys

from utils.Mapper import Mapper

# m = Mapper()
# os.makedirs('../data/big_dims_parts', exist_ok=True)
# os.makedirs('../data/small_dims_parts', exist_ok=True)
# m.split_to_blocks('../data/nn_embedding', '../data/big_dims_parts', '../data/small_dims_parts')
for i in range(805):
    with open('../data/big_dims_parts/' + str(i) + '.emb') as f:
        big = json.load(f)
    with open('../data/small_dims_parts/' + str(i) + '.emb') as f:
        small = json.load(f)
    for j in big:
        if j[19] is not None:
            assert j[22] is not None
    for j in small:
        assert j[22] is None
