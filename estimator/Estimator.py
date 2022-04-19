import importlib
import json
import math
import os

import sys

import torch.nn.functional as F
import optuna
import torch
import tqdm
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _get_observation_pairs, _split_observation_pairs
from torch import optim

from embedding.convert import *
from embedding.graph import *
from estimator.tmp import tmp
from utils.DatasetTransformer import Transformer


class Estimator:
    def __init__(self, embedding_width, generated_layers_count, train_dataloader, test_dataloader, device):
        self.support_ops = ["Conv",
                            "LeakyRelu",
                            "MaxPool",
                            "Flatten",
                            "Linear",
                            "Sigmoid",
                            "BatchNorm",
                            "Relu",
                            "Tanh",
                            "ConvTranspose"]
        self.big_dim_ops = [
            "Conv",
            "LeakyRelu",
            "MaxPool",
            "Sigmoid",
            "BatchNorm",
            "Relu",
            "Tanh",
            "ConvTranspose",
        ]
        self.small_dim_ops = [
            "Linear",
            "LeakyRelu",
            "Sigmoid",
            "Relu",
            "Tanh",
        ]
        self.embeddings = []
        self.device = device
        self.accuracy_non_zero_count = 0
        self.error_happened_count = 0
        self.first_in_shape = [1, 28, 28]
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.embedding_width = embedding_width
        self.generated_layers_count = generated_layers_count
        self.study = self.__pre_train()

        values, scores = _get_observation_pairs(self.study, ['big_dims_ops_count'], False, False)
        indices_below, indices_above = _split_observation_pairs(scores, self.study.sampler._gamma(len(scores)))
        self.good_indices = indices_below
        self.bad_indices = indices_above
        self.transformer = Transformer(embedding_width, len(self.embeddings[0]))
        self.transformer.transform_embeddings(self.embeddings)
        self.good_center = [[0.0 for _ in range(embedding_width)] for _ in range(len(self.embeddings[0]))]
        self.bad_center = [[0.0 for _ in range(embedding_width)] for _ in range(len(self.embeddings[0]))]
        for ind in self.good_indices:
            for i in range(len(self.embeddings[ind])):
                for j in range(embedding_width):
                    self.good_center[i][j] += self.embeddings[ind][i][j]
        for ind in self.bad_indices:
            for i in range(len(self.embeddings[ind])):
                for j in range(embedding_width):
                    self.bad_center[i][j] += self.embeddings[ind][i][j]
        for i in range(len(self.embeddings[0])):
            for j in range(embedding_width):
                self.good_center[i][j] /= len(self.good_indices)
                self.bad_center[i][j] /= len(self.bad_indices)

    # cuz I am generating MNIST classifiers the last part of a network should be like
    # Flatten -> Linear(10) -> LogSoftMax
    # So, generate it manually
    def __add_last_layer(self, in_shape, embedding, m):
        flatten = [None] * m
        linear = [None] * m
        log_softmax = [None] * m

        flatten[attribute_to_pos['op']] = node_to_ops['Flatten']

        if len(in_shape) == 3:
            flatten[5] = 1
            flatten[20] = 1  # batch_size
            flatten[21] = in_shape[1] * in_shape[2]
            in_shape = [1, in_shape[0] * in_shape[1] * in_shape[2]]
        if len(in_shape) == 2:
            flatten[5] = 1
            flatten[20] = 1  # batch_size
            flatten[21] = in_shape[1]

        linear[attribute_to_pos['op']] = node_to_ops['Linear']

        linear[0] = 1.0
        out_channel = 10
        linear[20] = 1
        linear[21] = out_channel

        in_shape = [1, out_channel]

        log_softmax[attribute_to_pos['op']] = node_to_ops['LogSoftmax']

        log_softmax[5] = 1
        log_softmax[20] = in_shape[0]
        log_softmax[21] = in_shape[1]

        embedding.append(flatten)
        embedding.append(linear)
        embedding.append(log_softmax)

    def objective(self, trial: Trial):
        n = self.generated_layers_count
        m = self.embedding_width
        embedding = []
        for i in range(n):
            embedding.append([None] * m)
        # first generate a graph structure
        big_dims_ops = trial.suggest_int("big_dims_ops_count", 1, min(5, n))
        for i in range(big_dims_ops):
            op = trial.suggest_categorical("big_op_" + str(i), self.big_dim_ops)
            embedding[i][attribute_to_pos['op']] = node_to_ops[op]
        for i in range(big_dims_ops, n):
            op = trial.suggest_categorical("small_op_" + str(i), self.small_dim_ops)
            embedding[i][attribute_to_pos['op']] = node_to_ops[op]

        # insert Flatten between big_dims_ops and small_dims_ops
        flatten = [None] * m
        flatten[attribute_to_pos['op']] = node_to_ops['Flatten']
        embedding.insert(big_dims_ops, flatten)

        in_shape = self.first_in_shape
        for i in range(len(embedding)):
            op = embedding[i][attribute_to_pos['op']]
            # CONV
            if op == 0:
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("conv_dilation_" + str(i), 1, 3)
                embedding[i][6] = dilation
                embedding[i][7] = dilation

                groups = 1
                embedding[i][13] = groups

                kernel_shape = trial.suggest_int("conv_kernel_shape_" + str(i), 2, 5)
                embedding[i][15] = kernel_shape
                embedding[i][16] = kernel_shape

                pads = trial.suggest_int("conv_pad_" + str(i), 0, 2)
                embedding[i][24] = pads
                embedding[i][25] = pads
                embedding[i][26] = pads
                embedding[i][27] = pads

                strides = trial.suggest_int("conv_stride_" + str(i), 1, 3)
                embedding[i][40] = strides
                embedding[i][41] = strides

                out_channel = trial.suggest_int("conv_out_channel_" + str(i), 1, 28)
                embedding[i][20] = 1  # batch_size
                embedding[i][21] = out_channel
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                embedding[i][22] = h_out
                embedding[i][23] = w_out
                in_shape = [out_channel, h_out, w_out]
            # LEAKY RELU
            if op == 1:
                if len(in_shape) == 2:
                    embedding[i][20] = in_shape[0]
                    embedding[i][21] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][20] = 1  # batch size
                    embedding[i][21] = in_shape[0]
                    embedding[i][22] = in_shape[1]
                    embedding[i][23] = in_shape[2]

                negative_slope = trial.suggest_categorical("leaky_rely_negative_slope_" + str(i), [1e-3, 1e-2, 1e-1])
                embedding[i][0] = negative_slope
                # SHAPE DOESN'T CHANGE
            # MAX POOL
            if op == 2:
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("maxpool_dilation_" + str(i), 1, 3)
                embedding[i][6] = dilation
                embedding[i][7] = dilation

                kernel_shape = trial.suggest_int("maxpool_kernel_shape_" + str(i), 2, 5)
                embedding[i][15] = kernel_shape
                embedding[i][16] = kernel_shape

                pads = trial.suggest_int("maxpool_pad_" + str(i), 0, 2)
                embedding[i][24] = pads
                embedding[i][25] = pads
                embedding[i][26] = pads
                embedding[i][27] = pads

                strides = trial.suggest_int("maxpool_stride_" + str(i), 1, 3)
                embedding[i][40] = strides
                embedding[i][41] = strides

                embedding[i][20] = 1  # batch_size
                embedding[i][21] = in_shape[0]
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                embedding[i][22] = h_out
                embedding[i][23] = w_out
                in_shape = [in_shape[0], h_out, w_out]
            # FLATTEN
            if op == 3:
                if len(in_shape) == 3:
                    embedding[i][5] = 1  # axis
                    embedding[i][20] = 1  # batch_size
                    embedding[i][21] = in_shape[1] * in_shape[2]
                    in_shape = [1, in_shape[0] * in_shape[1] * in_shape[2]]
                if len(in_shape) == 2:
                    embedding[i][5] = 1  # axis
                    embedding[i][20] = 1  # batch_size
                    embedding[i][21] = in_shape[1]
            # LINEAR
            if op == 4:
                embedding[i][0] = 1.0
                out_channel = trial.suggest_int("linear_out_channel_" + str(i), 1, 256)
                embedding[i][20] = 1
                embedding[i][21] = out_channel

                in_shape = [in_shape[0], out_channel]
            # SIGMOID
            if op == 5:
                if len(in_shape) == 2:
                    embedding[i][20] = in_shape[0]
                    embedding[i][21] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][20] = 1  # batch size
                    embedding[i][21] = in_shape[0]
                    embedding[i][22] = in_shape[1]
                    embedding[i][23] = in_shape[2]
                # SHAPE DOESN'T CHANGE
            # BATCH NORM
            if op == 6:
                embedding[i][20] = 1  # batch size
                embedding[i][21] = in_shape[0]
                embedding[i][22] = in_shape[1]
                embedding[i][23] = in_shape[2]

                epsilon = trial.suggest_categorical("batch_norm_epsilon_" + str(i), [1e-6, 1e-5, 1e-4])
                embedding[i][12] = epsilon

                momentum = 0.9
                embedding[i][18] = momentum

                # SHAPE DOESN'T CHANGE
            # RELU
            if op == 7:
                if len(in_shape) == 2:
                    embedding[i][20] = in_shape[0]
                    embedding[i][21] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][20] = 1  # batch size
                    embedding[i][21] = in_shape[0]
                    embedding[i][22] = in_shape[1]
                    embedding[i][23] = in_shape[2]
                # SHAPE DOESN'T CHANGE
            # Add
            if op == 8:
                continue
            # GlobalAveragePool
            if op == 9:
                continue
            # AveragePool
            if op == 10:
                continue
            # Concat
            if op == 11:
                continue
            # Pad
            if op == 12:
                continue
            # ReduceMean
            if op == 13:
                continue
            # Tanh
            if op == 14:
                if len(in_shape) == 2:
                    embedding[i][20] = in_shape[0]
                    embedding[i][21] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][20] = 1  # batch size
                    embedding[i][21] = in_shape[0]
                    embedding[i][22] = in_shape[1]
                    embedding[i][23] = in_shape[2]
                # SHAPE DOESN'T CHANGE
            # ConvTranspose
            if op == 15:
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("conv_transpose_dilation_" + str(i), 1, 3)
                embedding[i][6] = dilation
                embedding[i][7] = dilation

                groups = 1
                embedding[i][13] = groups

                kernel_shape = trial.suggest_int("conv_transpose_shape_" + str(i), 2, 5)
                embedding[i][15] = kernel_shape
                embedding[i][16] = kernel_shape

                pads = trial.suggest_int("conv_transpose_pad_" + str(i), 0, 2)
                embedding[i][24] = pads
                embedding[i][25] = pads
                embedding[i][26] = pads
                embedding[i][27] = pads

                strides = trial.suggest_int("conv_transpose_stride_" + str(i), 1, 3)
                embedding[i][40] = strides
                embedding[i][41] = strides

                out_channel = trial.suggest_int("conv_transpose_out_channel_" + str(i), 1, 28)
                embedding[i][20] = 1  # batch_size
                embedding[i][21] = out_channel
                h_out = (h_in - 1) * strides - 2 * pads + dilation * (kernel_shape - 1) + 1
                w_out = (w_in - 1) * strides - 2 * pads + dilation * (kernel_shape - 1) + 1
                embedding[i][22] = h_out
                embedding[i][23] = w_out

                in_shape = [out_channel, h_out, w_out]
        self.__add_last_layer(in_shape, embedding, m)
        for i in range(len(embedding)):
            # generate bamboo
            if i != len(embedding) - 1:
                embedding[i][ATTRIBUTES_POS_COUNT] = 1
                embedding[i][ATTRIBUTES_POS_COUNT + 1] = i + 1
            else:
                embedding[i][ATTRIBUTES_POS_COUNT] = 0
        return self.__get_quality_from_embedding(embedding)

    # TODO this function has to create network from embedding and return some metrics of quality
    # TODO for example accuracy
    def __get_quality_from_embedding(self, embedding):
        self.embeddings.append(embedding)
        graph = None
        try:
            graph = NeuralNetworkGraph.get_graph(embedding)
            print("GRAPH GENERATED")
        except Exception:
            return -1e6
        try:
            filepath = './tmp/tmp.py'
            model_name = 'Tmp'
            folders = os.path.dirname(filepath)
            os.makedirs(folders, exist_ok=True)
            Converter(graph, filepath=filepath, model_name=model_name)
            print("MODEL GENERATED")
        except Exception:
            return -1e3
        try:
            importlib.reload(tmp)

            model = tmp.Tmp().to(self.device)
            print('MODEL CREATED')
            # train model
            model.train()
            n_epoch = 5

            # mb configure optimizer here
            print('MODEL TRAINING STARTED')
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            for epoch in range(n_epoch):
                for i, (data, target) in enumerate(tqdm.tqdm(self.train_dataloader)):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data).to(self.device)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
            print('MODEL TRAINING FINISHED')
            # testing and calculating accuracy
            model.eval()
            correct = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(tqdm.tqdm(self.test_dataloader)):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data).to(self.device)
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            # return accuracy
            accuracy = 100. * correct.item() / len(self.test_dataloader.dataset)
            # os.makedirs('./estimator_generated_embeddings', exist_ok=True)
            # with open('./estimator_generated_embeddings/' + str(self.accuracy_non_zero_count) + '_' + str(
            #         accuracy) + '.txt', 'w+') as f:
            #     f.write(json.dumps(embedding))
            # self.accuracy_non_zero_count += 1
            return accuracy
        except Exception as e:
            # os.makedirs('./estimator_failed_generated_embeddings', exist_ok=True)
            # with open('./estimator_failed_generated_embeddings/' + str(self.error_happened_count) + '.txt', 'w+') as f:
            #     f.write(json.dumps(embedding))
            # self.error_happened_count += 1
            # print(str(e))
            return -1e2

    def __pre_train(self):
        sampler = TPESampler()
        study = optuna.create_study(sampler=sampler, direction='maximize')
        # should do some limitation of pre train here for example count of trials or time limit
        n_trials = 100
        study.optimize(self.objective, n_trials=n_trials)
        return study

    # Euclidean metrics
    @staticmethod
    def __metrics(a, b):
        inf = 1e3  # dist between None and non-None
        if len(a) != len(b):
            raise Exception("Embeddings should be same size")
        if len(a) == 0:
            return 0.0
        res = 0.0
        for i in range(len(a)):
            if len(a[i]) != len(b[i]):
                raise Exception("Embeddings have different sizes at string: " + str(i))
            for j in range(len(a[i])):
                if (a[i][j] is None) and (b[i][j] is None):
                    res += 0.0
                elif (a[i][j] is None) or (b[i][j] is None):
                    res += inf
                else:
                    res += (a[i][j] - b[i][j]) ** 2
        return math.sqrt(res)

    def check(self, embedding):
        return self.__metrics(self.good_center, embedding) > self.__metrics(self.bad_center, embedding)
