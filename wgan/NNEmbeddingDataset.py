import json
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class NNEmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])

    def __getitem__(self, index: int) -> T_co:
        emb_file = os.path.join(self.root_dir, str(index) + '.emb')
        with open(emb_file) as f:
            embedding = json.load(f)
        return torch.tensor(embedding)
