import json
import math
import os
import sys

import optuna
import torch
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _get_observation_pairs, _split_observation_pairs
from torch import optim

from embedding.convert import *
from embedding.graph import *


class Estimator:
    def __init__(self, embedding_width, embedding_height, train_dataloader, test_dataloader, loss):
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
        self.first_in_shape = [1, 28, 28]
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss = loss
        self.good_center = [0.0] * embedding_width
        self.bad_center = [0.0] * embedding_width

        self.embedding_width = embedding_width
        self.embedding_height = embedding_height
        self.study = self.__pre_train()

        # TODO CHANGE EVERYTHING BELOW TOO
        # values, scores = _get_observation_pairs(self.study, [self.__get_param_name(0, 0)], False, False)
        # indices_below, indices_above = _split_observation_pairs(scores, self.study.sampler._gamma(len(scores)))
        # trials = self.study.get_trials()
        # self.points = []
        # # True = good, False = bad
        # for ind in indices_below:
        #     cur_point = []
        #     for i in range(self.embedding_height):
        #         cur_line = []
        #         for j in range(self.embedding_width):
        #             cur_line.append(float(trials[ind].params[self.__get_param_name(i, j)]))
        #             self.good_center[j] += cur_line[j]
        #         cur_point.append(cur_line)
        #     self.points.append((cur_point, True))
        # for ind in indices_above:
        #     cur_point = []
        #     for i in range(self.embedding_height):
        #         cur_line = []
        #         for j in range(self.embedding_width):
        #             cur_line.append(float(trials[ind].params[self.__get_param_name(i, j)]))
        #             self.bad_center[j] += cur_line[j]
        #         cur_point.append(cur_line)
        #     self.points.append((cur_point, False))
        # good_cnt = len(indices_below)
        # bad_cnt = len(indices_above)
        # for i in range(self.embedding_width):
        #     self.good_center[i] /= good_cnt
        #     self.bad_center[i] /= bad_cnt

    def objective(self, trial: Trial):
        n = trial.suggest_int("embedding_height", 1, self.embedding_height)
        # m = trial.suggest_int("embedding_width", 1, self.embedding_width)
        m = self.embedding_width
        embedding = []
        for i in range(n):
            embedding.append([None] * m)
        # first generate a graph structure
        for i in range(n):
            op = trial.suggest_categorical("op_" + str(i), self.support_ops)
            embedding[i][attribute_to_pos['op']] = node_to_ops[op]
            # generate bamboo
            if i < n - 1:
                embedding[i][ATTRIBUTES_POS_COUNT] = 1
                embedding[i][ATTRIBUTES_POS_COUNT + 1] = i + 1
            else:
                embedding[i][ATTRIBUTES_POS_COUNT] = 0
        in_shape = self.first_in_shape
        for i in range(n):
            # TODO op is int now, change it in ifs below
            op = embedding[i][ATTRIBUTES_POS_COUNT - 1]
            # CONV
            if op == 0:
                if len(in_shape) != 3:
                    return -1e9
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("conv_dilation_" + str(i), 1, 3)
                embedding[i][0] = dilation
                embedding[i][1] = dilation

                groups = 1
                embedding[i][2] = groups

                kernel_shape = trial.suggest_int("conv_kernel_shape_" + str(i), 2, 5)
                embedding[i][3] = kernel_shape
                embedding[i][4] = kernel_shape

                pads = trial.suggest_int("conv_pad_" + str(i), 0, 2)
                embedding[i][5] = pads
                embedding[i][6] = pads
                embedding[i][7] = pads
                embedding[i][8] = pads

                strides = trial.suggest_int("conv_stride_" + str(i), 1, 3)
                embedding[i][9] = strides
                embedding[i][10] = strides

                out_channel = trial.suggest_int("conv_out_channel_" + str(i), 1, 28)
                embedding[i][11] = 1  # batch_size
                embedding[i][12] = out_channel
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                embedding[i][13] = h_out
                embedding[i][14] = w_out
                in_shape = [out_channel, h_out, w_out]
            # LEAKY RELU
            if op == 1:
                if len(in_shape) == 2:
                    embedding[i][17] = in_shape[0]
                    embedding[i][18] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][11] = 1  # batch size
                    embedding[i][12] = in_shape[0]
                    embedding[i][13] = in_shape[1]
                    embedding[i][14] = in_shape[2]

                negative_slope = trial.suggest_categorical("leaky_rely_negative_slope_" + str(i), [1e-3, 1e-2, 1e-1])
                embedding[i][15] = negative_slope
                # SHAPE DOESN'T CHANGE
            # MAX POOL
            if op == 2:
                if len(in_shape) != 3:
                    return -1e9
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("maxpool_dilation_" + str(i), 1, 3)
                embedding[i][0] = dilation
                embedding[i][1] = dilation

                kernel_shape = trial.suggest_int("maxpool_kernel_shape_" + str(i), 2, 5)
                embedding[i][3] = kernel_shape
                embedding[i][4] = kernel_shape

                pads = trial.suggest_int("maxpool_pad_" + str(i), 0, 2)
                embedding[i][5] = pads
                embedding[i][6] = pads
                embedding[i][7] = pads
                embedding[i][8] = pads

                strides = trial.suggest_int("maxpool_stride_" + str(i), 1, 3)
                embedding[i][9] = strides
                embedding[i][10] = strides

                embedding[i][11] = 1  # batch_size
                embedding[i][12] = in_shape[0]
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_shape - 1) - 1) / strides + 1)
                embedding[i][13] = h_out
                embedding[i][14] = w_out
                in_shape = [in_shape[0], h_out, w_out]
            # FLATTEN
            if op == 3:
                if len(in_shape) != 3:
                    return -1e9

                embedding[i][16] = 1
                embedding[i][17] = 1  # batch_size
                embedding[i][18] = in_shape[1] * in_shape[2]
                in_shape = [1, in_shape[1] * in_shape[2]]
            # LINEAR
            if op == 4:
                if len(in_shape) != 2:
                    return -1e9

                embedding[i][15] = 1.0
                out_channel = trial.suggest_int("linear_out_channel_" + str(i), 1, 256)
                embedding[i][17] = 1
                embedding[i][18] = out_channel
                embedding[i][19] = 1.0
                embedding[i][20] = 1

                in_shape = [1, out_channel]
            # SIGMOID
            if op == 5:
                if len(in_shape) == 2:
                    embedding[i][17] = in_shape[0]
                    embedding[i][18] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][11] = 1  # batch size
                    embedding[i][12] = in_shape[0]
                    embedding[i][13] = in_shape[1]
                    embedding[i][14] = in_shape[2]
                # SHAPE DOESN'T CHANGE
            # BATCH NORM
            if op == 6:
                if len(in_shape) != 3:
                    return -1e9

                embedding[i][11] = 1  # batch size
                embedding[i][12] = in_shape[0]
                embedding[i][13] = in_shape[1]
                embedding[i][14] = in_shape[2]

                epsilon = trial.suggest_categorical("batch_norm_epsilon_" + str(i), [1e-6, 1e-5, 1e-4])
                embedding[i][21] = epsilon

                momentum = 0.9
                embedding[i][22] = momentum

                # SHAPE DOESN'T CHANGE
            # RELU
            if op == 7:
                if len(in_shape) == 2:
                    embedding[i][17] = in_shape[0]
                    embedding[i][18] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][11] = 1  # batch size
                    embedding[i][12] = in_shape[0]
                    embedding[i][13] = in_shape[1]
                    embedding[i][14] = in_shape[2]
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
                    embedding[i][17] = in_shape[0]
                    embedding[i][18] = in_shape[1]

                if len(in_shape) == 3:
                    embedding[i][11] = 1  # batch size
                    embedding[i][12] = in_shape[0]
                    embedding[i][13] = in_shape[1]
                    embedding[i][14] = in_shape[2]
                # SHAPE DOESN'T CHANGE
            # ConvTranspose
            if op == 15:
                if len(in_shape) != 3:
                    return -1e9
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = trial.suggest_int("conv_transpose_dilation_" + str(i), 1, 3)
                embedding[i][0] = dilation
                embedding[i][1] = dilation

                groups = 1
                embedding[i][2] = groups

                kernel_shape = trial.suggest_int("conv_transpose_shape_" + str(i), 2, 5)
                embedding[i][3] = kernel_shape
                embedding[i][4] = kernel_shape

                pads = trial.suggest_int("conv_transpose_pad_" + str(i), 0, 2)
                embedding[i][5] = pads
                embedding[i][6] = pads
                embedding[i][7] = pads
                embedding[i][8] = pads

                strides = trial.suggest_int("conv_transpose_stride_" + str(i), 1, 3)
                embedding[i][9] = strides
                embedding[i][10] = strides

                out_channel = trial.suggest_int("conv_transpose_out_channel_" + str(i), 1, 28)
                embedding[i][11] = 1  # batch_size
                embedding[i][12] = out_channel
                h_out = (h_in - 1) * strides - 2 * pads + dilation * (kernel_shape - 1) + 1
                w_out = (w_in - 1) * strides - 2 * pads + dilation * (kernel_shape - 1) + 1
                embedding[i][13] = h_out
                embedding[i][14] = w_out

                in_shape = [out_channel, h_out, w_out]

        return self.__get_quality_from_embedding(embedding)

    # TODO this function has to create network from embedding and return some metrics of quality
    # TODO for example accuracy
    def __get_quality_from_embedding(self, embedding):
        graph = None
        try:
            graph = NeuralNetworkGraph.get_graph(embedding)
            print("GRAPH GENERATED")
        except Exception:
            return -1e6
        try:
            filepath = 'tmp/tmp.py'
            model_name = 'Tmp'
            folders = os.path.dirname(filepath)
            os.makedirs(folders, exist_ok=True)
            Converter(graph, filepath=filepath, model_name=model_name)
            os.makedirs('./graph_generated_embeddings', exist_ok=True)
            with open('./graph_generated_embeddings/good_embedding.txt', 'w+') as f:
                f.write(json.dumps(embedding))
            print("MODEL GENERATED")
        except Exception:
            return -1e3
        return 0
        # if generated model throws exception while training or testing = generated embedding is very bad
        # try:
        #     # generate a file with Pytorch realization of embedded network
        #     graph = NeuralNetworkGraph.get_graph(embedding)
        #     print("Graph generated")
        #
        #     # os.makedirs('./graph_generated_embeddings', exist_ok=True)
        #     # with open('./graph_generated_embeddings/good_embedding.txt', 'w+') as f:
        #     #     f.write(json.dumps(embedding))

            # filepath = 'tmp/tmp.py'
            # model_name = 'Tmp'
            # folders = os.path.dirname(filepath)
            # os.makedirs(folders, exist_ok=True)
            # Converter(graph, filepath=filepath, model_name=model_name)
            # print("Model generated")
        #     sys.path.append(folders)
        #
        #     model = Tmp()
        #
        #     # train model
        #     model.train()
        #     n_epoch = 10
        #
        #     # mb configure optimizer here
        #     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        #     for epoch in range(n_epoch):
        #         for data, target in self.train_dataloader:
        #             optimizer.zero_grad()
        #             output = model(data)
        #             loss = self.loss(output, target)
        #             loss.backward()
        #             optimizer.step()
        #
        #     # testing and calculating accuracy
        #     model.eval()
        #     correct = 0
        #     with torch.no_grad():
        #         for data, target in self.test_dataloader:
        #             output = model(data)
        #         pred = output.data.max(1, keepdim=True)[1]
        #         correct += pred.eq(target.data.view_as(pred)).sum()
        #     # return accuracy
        #     return 100. * correct / len(self.test_dataloader.dataset)
        # except Exception as e:
        #     print(str(e))
        #     return -1e9

    def __pre_train(self):
        sampler = TPESampler()
        study = optuna.create_study(sampler=sampler, direction='maximize')
        # should do some limitation of pre train here for example count of trials or time limit
        n_trials = 1000
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
