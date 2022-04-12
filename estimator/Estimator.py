import math
import os
import sys

import optuna
import torch
import torch.nn.functional as F
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _get_observation_pairs, _split_observation_pairs
from torch import optim

from embedding.convert import Converter
from embedding.graph import NeuralNetworkGraph


class Estimator:
    @staticmethod
    def __get_param_name(i, j, suffix):
        return "x_" + str(i) + "_" + str(j) + "_" + suffix

    def __init__(self, embedding_width, embedding_height, train_dataloader, test_dataloader, optimizer, loss):
        self.inf = math.inf
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss = loss

        self.embedding_width = embedding_width
        self.embedding_height = embedding_height
        self.study = self.pre_train()
        values, scores = _get_observation_pairs(self.study, [self.__get_param_name(0, 0, "null")], False, False)
        indices_below, indices_above = _split_observation_pairs(scores, self.study.sampler._gamma(len(scores)))
        trials = self.study.get_trials()
        self.points = []
        # True = good, False = bad
        for ind in indices_below:
            cur_point = []
            for i in range(self.embedding_height):
                cur_line = []
                for j in range(self.embedding_width):
                    if trials[ind].params[self.__get_param_name(i, j, "null")] == "null":
                        cur_line.append(None)
                    else:
                        cur_line.append(float(trials[ind].params[self.__get_param_name(i, j, "")]))
                cur_point.append(cur_line)
            self.points.append((cur_point, True))
        for ind in indices_above:
            cur_point = []
            for i in range(self.embedding_height):
                cur_line = []
                for j in range(self.embedding_width):
                    if trials[ind].params[self.__get_param_name(i, j, "null")] == "null":
                        cur_line.append(None)
                    else:
                        cur_line.append(float(trials[ind].params[self.__get_param_name(i, j, "")]))
                cur_point.append(cur_line)
            self.points.append((cur_point, False))

    def objective(self, trial: Trial):
        n = trial.suggest_int("embedding_height", 1, self.embedding_height)
        m = trial.suggest_int("embedding_width", 1, self.embedding_width)
        embedding = []
        for i in range(n):
            cur_line = []
            for j in range(m):
                # TODO decide this values
                # TODO low and high should be chosen with a margin cuz optuna doesn't search in bounds area
                low = -100.0
                high = 100.0
                step = 0.1
                is_null = trial.suggest_categorical(self.__get_param_name(i, j, "null"), ["null", "not_null"])
                if is_null == "null":
                    x_i = None
                else:
                    x_i = trial.suggest_float(self.__get_param_name(i, j, ""), low, high, step=step)
                # mb use another distribution for parameters of embedding
                cur_line.append(x_i)
            embedding.append(cur_line)
        return self.get_quality_from_embedding(embedding)

    # TODO this function has to create network from embedding and return some metrics of quality
    # TODO for example accuracy
    def get_quality_from_embedding(self, embedding):
        graph = NeuralNetworkGraph.get_graph(embedding)
        filepath = 'tmp/tmp.py'
        model_name = 'Tmp'
        folders = os.path.dirname(filepath)
        os.makedirs(folders, exist_ok=True)
        Converter(graph, filepath=filepath, model_name=model_name)
        sys.path.append(folders)
        # if generated model throws exception while training or testing = generated embedding is very bad
        try:
            model = Tmp()

            # train model
            model.train()
            n_epoch = 10

            for epoch in range(n_epoch):
                for data, target in self.train_dataloader:
                    self.optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss(output, target)
                    loss.backward()
                    # mb should re-create optimizer for every invocation of get_quality_from_embedding method
                    self.optimizer.step()

            # testing and calculating accuracy
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in self.test_dataloader:
                    output = model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            # return accuracy
            return 100. * correct / len(self.test_dataloader.dataset)
        except Exception:
            return -self.inf

    def pre_train(self):
        sampler = TPESampler()
        study = optuna.create_study(sampler=sampler, direction='maximize')
        # should do some limitation of pre train here for example count of trials or time limit
        n_trials = 1000
        study.optimize(self.objective, n_trials=n_trials)
        return study

    # Euclidean metrics
    @staticmethod
    def __metrics(a, b):
        inf = 1e9  # dist between None and non-None
        if len(a) != len(b):
            raise Exception("Embeddings should be same size")
        if len(a) == 0:
            return 0
        res = 0
        for i in range(len(a)):
            if len(a[i]) != len(b[i]):
                raise Exception("Embeddings have different sizes at string: " + str(i))
            for j in range(len(a[i])):
                if (a[i][j] is None) or (b[i][j] is None):
                    res += inf
                else:
                    res += (a[i][j] - b[i][j]) ** 2
        return math.sqrt(res)

    def check(self, embedding):
        # take some radius for k nearest neighbours method
        r = 1337
        count_near_good = 0
        count_near_bad = 0
        for p in self.points:
            dist = self.__metrics(embedding, p)
            if dist <= r:
                # is good
                if p[1]:
                    count_near_good += 1
                else:
                    count_near_bad += 1
        return count_near_good > count_near_bad
