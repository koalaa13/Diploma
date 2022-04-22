import json
import math
import os


class Transformer:
    def __init__(self, embedding_width, embedding_height):
        self.padding_elem = 0.0
        self.null_as_number = -1.0
        self.embedding_width = embedding_width
        self.embedding_height = embedding_height
        self.mns = [math.inf] * embedding_width
        self.mxs = [-math.inf] * embedding_width

    def transform_embeddings(self, embeddings):
        for i in range(len(embeddings)):
            self.transform_embedding(embeddings[i])
        for k in range(len(embeddings)):
            for i in range(self.embedding_height):
                for j in range(self.embedding_width):
                    self.mns[j] = min(self.mns[j], embeddings[k][i][j])
                    self.mxs[j] = max(self.mxs[j], embeddings[k][i][j])
        for k in range(len(embeddings)):
            for i in range(self.embedding_height):
                for j in range(self.embedding_width):
                    if self.mxs[j] - self.mns[j] != 0:
                        embeddings[k][i][j] = 2 * (embeddings[k][i][j] - self.mns[j]) / (self.mxs[j] - self.mns[j]) - 1
                        assert -1.0 <= embeddings[k][i][j] <= 1.0
                    else:
                        embeddings[k][i][j] = self.mns[j]

    def transform_embedding(self, embedding):
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

    def de_transform_embedding(self, embedding):
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                embedding[i][j] = (embedding[i][j] + 1) * (self.mxs[j] - self.mns[j]) / 2 + self.mns[j]
                if embedding[i][j] < 0.0:
                    embedding[i][j] = None

    def transform_dataset(self, dataset_folder, transformed_dataset_folder):
        embeddings = []
        for file in os.listdir(dataset_folder):
            with open(os.path.join(dataset_folder, file)) as f:
                embedding = json.load(f)
            embeddings.append(embedding)
        self.transform_embeddings(embeddings)
        for i in range(len(embeddings)):
            with open(os.path.join(transformed_dataset_folder, str(i) + '.emb'), 'w+') as f:
                f.write(json.dumps(embeddings[i]))
