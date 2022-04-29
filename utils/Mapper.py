import copy
import json
import math
import os

from embedding.graph import attribute_to_pos, NODE_EMBEDDING_DIMENSION, ATTRIBUTES_POS_COUNT, node_to_ops


class Mapper:
    def __init__(self):
        self.corteges = None
        self.op_to_indices = {
            0: [6, 13, 15, 24, 40, 21],
            1: [0],
            2: [6, 15, 24, 40],
            3: [],
            4: [21],
            5: [],
            6: [12],
            7: [],
            14: [],
            15: [6, 13, 15, 24, 40, 21],
            22: []
        }
        self.small_embedding_width = -math.inf
        for k, v in self.op_to_indices.items():
            self.small_embedding_width = max(self.small_embedding_width, len(v))

    def __get_small_row(self, row):
        op = row[attribute_to_pos['op']]
        cur_row = [op]
        for ind in self.op_to_indices[op]:
            cur_row.append(row[ind])
        while len(cur_row) < self.small_embedding_width:
            cur_row.append(None)
        return cur_row

    @staticmethod
    def __get_big_and_small(embedding):
        first_2d = -1
        for i in range(len(embedding)):
            if embedding[i][19] == node_to_ops['Flatten']:
                first_2d = i
                break
        if first_2d == -1:
            first_2d = 5
        return copy.deepcopy(embedding[0:first_2d]), copy.deepcopy(embedding[first_2d + 1:11])

    def split_to_blocks(self, src, dst_big, dst_small):
        for file in os.listdir(src):
            with open(os.path.join(src, file)) as f:
                embedding = json.load(f)
            big, small = self.__get_big_and_small(embedding)
            while len(big) < 5:
                big.append([None] * len(big[0]))
            while len(small) < 9:
                small.append([None] * len(small[0]))
            with open(os.path.join(dst_big, file), 'w+') as f:
                f.write(json.dumps(big))
            with open(os.path.join(dst_small, file), 'w+') as f:
                f.write(json.dumps(small))

    def map_to_super_small_embedding(self, src, dst):
        corteges = set()
        for file in os.listdir(src):
            with open(os.path.join(src, file)) as f:
                embedding = json.load(f)
            for row in embedding:
                if row[attribute_to_pos['op']] is None:
                    break
                small_row = tuple(self.__get_small_row(row))
                corteges.add(small_row)
        corteges = sorted(list(corteges))
        self.corteges = corteges
        for file in os.listdir(src):
            super_small_embedding = []
            with open(os.path.join(src, file)) as f:
                embedding = json.load(f)
            for row in embedding:
                if row[attribute_to_pos['op']] is None:
                    super_small_embedding.append([None])
                else:
                    super_small_embedding.append([corteges.index(tuple(self.__get_small_row(row)))])
            with open(os.path.join(dst, file), 'w+') as f:
                f.write(json.dumps(super_small_embedding))

    def de_map_from_super_small_embedding(self, embedding, in_shape):
        res = []
        print('IN SHAPE = ' + str(in_shape))
        for jj in range(len(embedding)):
            cur_row = [None] * NODE_EMBEDDING_DIMENSION
            ind = int(embedding[jj][0] + 0.5)
            cort = self.corteges[ind]
            cur_row[attribute_to_pos['op']] = cort[0]
            if cort[0] == 0:  # CONV
                if len(in_shape) != 3:
                    raise Exception('Incorrect shape')
                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = cort[1]
                cur_row[6] = dilation
                cur_row[7] = dilation

                groups = cort[2]
                cur_row[13] = groups

                kernel_size = cort[3]
                cur_row[15] = kernel_size
                cur_row[16] = kernel_size

                pads = cort[4]
                cur_row[24] = pads
                cur_row[25] = pads
                cur_row[26] = pads
                cur_row[27] = pads

                strides = cort[5]
                cur_row[40] = strides
                cur_row[41] = strides

                out_channel = cort[6]
                cur_row[20] = 1
                cur_row[21] = out_channel
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_size - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_size - 1) - 1) / strides + 1)
                cur_row[22] = h_out
                cur_row[23] = w_out

                in_shape = [out_channel, h_out, w_out]
            if cort[0] == 1:  # LEAKY RELU
                if len(in_shape) == 2:
                    cur_row[20] = in_shape[0]
                    cur_row[21] = in_shape[1]

                if len(in_shape) == 3:
                    cur_row[20] = 1  # batch size
                    cur_row[21] = in_shape[0]
                    cur_row[22] = in_shape[1]
                    cur_row[23] = in_shape[2]

                negative_slope = cort[1]
                cur_row[0] = negative_slope
            if cort[0] == 2:  # MAX POOL
                if len(in_shape) != 3:
                    raise Exception('Incorrect shape')

                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = cort[1]
                cur_row[6] = dilation
                cur_row[7] = dilation

                kernel_size = cort[2]
                cur_row[15] = kernel_size
                cur_row[16] = kernel_size

                pads = cort[3]
                cur_row[24] = pads
                cur_row[25] = pads
                cur_row[26] = pads
                cur_row[27] = pads

                strides = cort[4]
                cur_row[40] = strides
                cur_row[41] = strides

                cur_row[20] = 1
                cur_row[21] = in_shape[0]
                h_out = math.floor((h_in + 2 * pads - dilation * (kernel_size - 1) - 1) / strides + 1)
                w_out = math.floor((w_in + 2 * pads - dilation * (kernel_size - 1) - 1) / strides + 1)
                cur_row[22] = h_out
                cur_row[23] = w_out

                in_shape = [in_shape[0], h_out, w_out]
            if cort[0] == 3:  # FLATTEN
                if len(in_shape) == 3:
                    cur_row[5] = 1  # axis
                    cur_row[20] = 1  # batch_size
                    cur_row[21] = in_shape[1] * in_shape[2]
                    in_shape = [1, in_shape[0] * in_shape[1] * in_shape[2]]
                if len(in_shape) == 2:
                    cur_row[5] = 1  # axis
                    cur_row[20] = 1  # batch_size
                    cur_row[21] = in_shape[1]
            if cort[0] == 4:  # LINEAR
                if len(in_shape) != 2:
                    raise Exception('Incorrect shape')

                cur_row[0] = 1.0
                cur_row[20] = 1
                out_channel = cort[1]
                cur_row[21] = out_channel

                in_shape = [in_shape[0], out_channel]
                print('Linear out shape = ' + str(in_shape))
            if cort[0] == 5:  # SIGMOID
                if len(in_shape) == 2:
                    cur_row[20] = in_shape[0]
                    cur_row[21] = in_shape[1]

                if len(in_shape) == 3:
                    cur_row[20] = 1  # batch size
                    cur_row[21] = in_shape[0]
                    cur_row[22] = in_shape[1]
                    cur_row[23] = in_shape[2]
            if cort[0] == 6:  # BATCH NORM
                if len(in_shape) != 3:
                    raise Exception('Incorrect shape')

                cur_row[20] = 1  # batch size
                cur_row[21] = in_shape[0]
                cur_row[22] = in_shape[1]
                cur_row[23] = in_shape[2]

                epsilon = cort[1]
                cur_row[12] = epsilon

                momentum = 0.9
                cur_row[18] = momentum
            if cort[0] == 7:  # RELU
                if len(in_shape) == 2:
                    cur_row[20] = in_shape[0]
                    cur_row[21] = in_shape[1]

                if len(in_shape) == 3:
                    cur_row[20] = 1  # batch size
                    cur_row[21] = in_shape[0]
                    cur_row[22] = in_shape[1]
                    cur_row[23] = in_shape[2]
            if cort[0] == 14:  # TANH
                if len(in_shape) == 2:
                    cur_row[20] = in_shape[0]
                    cur_row[21] = in_shape[1]

                if len(in_shape) == 3:
                    cur_row[20] = 1  # batch size
                    cur_row[21] = in_shape[0]
                    cur_row[22] = in_shape[1]
                    cur_row[23] = in_shape[2]
            if cort[0] == 15:  # CONV TRANSPOSE
                if len(in_shape) != 3:
                    raise Exception('Incorrect shape')

                h_in = in_shape[1]
                w_in = in_shape[2]

                dilation = cort[1]
                cur_row[6] = dilation
                cur_row[7] = dilation

                groups = cort[2]
                cur_row[13] = groups

                kernel_size = cort[3]
                cur_row[15] = kernel_size
                cur_row[16] = kernel_size

                pads = cort[4]
                cur_row[24] = pads
                cur_row[25] = pads
                cur_row[26] = pads
                cur_row[27] = pads

                strides = cort[5]
                cur_row[40] = strides
                cur_row[41] = strides

                out_channel = cort[6]
                cur_row[20] = 1
                cur_row[21] = out_channel
                h_out = (h_in - 1) * strides - 2 * pads + dilation * (kernel_size - 1) + 1
                w_out = (w_in - 1) * strides - 2 * pads + dilation * (kernel_size - 1) + 1

                cur_row[22] = h_out
                cur_row[23] = w_out

                in_shape = [out_channel, h_out, w_out]
            if cort[0] == 22:  # LOGSOFTMAX
                cur_row[5] = 1
                if len(in_shape) == 2:
                    cur_row[20] = in_shape[0]
                    cur_row[21] = in_shape[1]

                if len(in_shape) == 3:
                    cur_row[20] = 1  # batch size
                    cur_row[21] = in_shape[0]
                    cur_row[22] = in_shape[1]
                    cur_row[23] = in_shape[2]
            res.append(cur_row)
        return res, in_shape
