import copy
import json
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class NNEmbeddingDataset(Dataset):
    def __init__(self, root_dir, embedding_width, embedding_height, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.embedding_width = embedding_width
        self.embedding_height = embedding_height
        self.padding_element = 0
        self.none_as_number = -1

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def __add_padding(self, embedding):
        res = []
        for i in range(self.embedding_height):
            if i < len(embedding):
                cur_row = copy.copy(embedding[i])
                while len(cur_row) < self.embedding_width:
                    cur_row.append(self.padding_element)
                res.append(cur_row)
            else:
                res.append([self.padding_element] * self.embedding_width)
        return res

    def __change_none_to_number(self, embedding):
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if embedding[i][j] is None:
                    embedding[i][j] = self.none_as_number
        return embedding

    def __getitem__(self, index) -> T_co:
        if torch.is_tensor(index):
            index = index.tolist()
        emb_file = os.path.join(self.root_dir, str(index) + '.emb')
        with open(emb_file) as f:
            embedding = json.load(f)
        return self.__change_none_to_number(self.__add_padding(embedding))
