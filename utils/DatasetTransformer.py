import json
import math
import os


class Transformer:
    def __init__(self, embedding_width, embedding_height):
        self.non_transformed_folder = "../data/nn_embedding"
        self.transformed_folder = "../data/nn_embedding_transformed"
        self.padding_elem = 0.0
        self.null_as_number = -1.0
        self.embedding_width = embedding_width
        self.embedding_height = embedding_height

    def transform(self):
        for file in os.listdir(self.non_transformed_folder):
            with open(os.path.join(self.non_transformed_folder, file)) as f:
                embedding = json.load(f)
            mx = -math.inf
            mn = -mx
            for i in range(len(embedding)):
                for j in range(len(embedding[i])):
                    if embedding[i][j] is None:
                        embedding[i][j] = self.null_as_number
                    else:
                        embedding[i][j] = float(embedding[i][j])
            for i in range(self.embedding_height):
                if i < len(embedding):
                    while len(embedding[i]) < self.embedding_width:
                        embedding[i].append(self.padding_elem)
                else:
                    embedding.append([self.padding_elem] * self.embedding_width)
            for i in range(len(embedding)):
                for j in range(len(embedding[i])):
                    mn = min(mn, embedding[i][j])
                    mx = max(mx, embedding[i][j])
            for i in range(len(embedding)):
                for j in range(len(embedding[i])):
                    embedding[i][j] = 2 * (embedding[i][j] - mn) / (mx - mn) - 1
            os.makedirs(self.transformed_folder, exist_ok=True)
            with open(os.path.join(self.transformed_folder, file), "w+") as f:
                print([embedding], file=f)
