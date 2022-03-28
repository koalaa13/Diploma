import math
from functools import partial

import optuna
from optuna import Trial, Study
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _get_observation_pairs, _split_observation_pairs
import numpy as np


# Chebyshev's metrics
def metrics(a, b):
    if len(a) != len(b):
        raise Exception("Vectors should be same size")
    res = math.fabs(a[0] - b[0])
    for i in range(len(a)):
        res = max(res, math.fabs(a[i] - b[i]))
    return res


class Estimator:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
        self.study = self.pre_train()
        values, scores = _get_observation_pairs(self.study, ["x0"], False, False)
        indices_below, indices_above = _split_observation_pairs(scores, self.study.sampler._gamma(len(scores)))
        trials = self.study.get_trials()
        self.goods = []
        for ind in indices_below:
            cur_point = []
            for i in range(self.embedding_size):
                cur_point.append(float(trials[ind].params["x" + str(i)]))
            self.goods.append(cur_point)

    # decide a shape of an embedding now it's just a vector
    # do something with null if we will have them in embeddings. for example optuna has conditionals values
    def objective(self, trial: Trial):
        embedding = []
        for i in range(self.embedding_size):
            # decide this values
            # low and high should be chosen with a margin cuz optuna doesn't search in bounds area
            low = -100.0
            high = 100.0
            step = 0.1
            # mb use another distribution for parameters of embedding
            x_i = trial.suggest_float("x" + str(i), low, high, step=step)
            embedding.append(x_i)
        return self.get_quality_from_embedding(embedding)

    # TODO this function has to create network from embedding and return some metrics of quality
    # TODO for example accuracy
    def get_quality_from_embedding(self, embedding):
        raise NotImplementedError

    def pre_train(self):
        sampler = TPESampler()
        study = optuna.create_study(sampler=sampler, direction='maximize')
        # should do some limitation of pre train here for example count of trials or time limit
        n_trials = 1000
        study.optimize(self.objective, n_trials=n_trials)
        return study

    def check(self, embedding):
        # TODO here should decide is embedding good or not without launching a net
        # TODO for example calc some metrics between self.goods and given embedding
        res = 0.0
        for i in self.goods:
            res += metrics(embedding, i)
        return res
