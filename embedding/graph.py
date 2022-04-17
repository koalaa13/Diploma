import hiddenlayer as hl
import networkx as nx

ATTRIBUTES_POS_COUNT = 37
NODE_EMBEDDING_DIMENSION = 100
NONE_REPLACEMENT = -1

node_to_ops = {
    "Conv": 0,
    "LeakyRelu": 1,
    "MaxPool": 2,
    "Flatten": 3,
    "Linear": 4,
    "Sigmoid": 5,
    "BatchNorm": 6,
    "Relu": 7,
    "Add": 8,
    "GlobalAveragePool": 9,
    "AveragePool": 10,
    "Concat": 11,
    "Pad": 12,
    "ReduceMean": 13,
    "Tanh": 14,
    "ConvTranspose": 15,
}

pads_to_mods = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}

ops_with_different_dims = ["output_shape", "pads"]

attribute_to_pos = {
    "dilations": [0, 1],
    "group": 2,
    "kernel_shape": [3, 4],
    "pads_4": [5, 6, 7, 8],
    "strides": [9, 10],
    "output_shape_4": [11, 12, 13, 14],
    "alpha": 15,
    "axis": 16,
    "output_shape_2": [17, 18],
    "beta": 19,
    "transB": 20,
    "epsilon": 21,
    "momentum": 22,
    "mode": 23,
    "pads_8": [24, 25, 26, 27, 28, 29, 30, 31],
    "value": 32,
    "axes": [33, 34],
    "keepdims": 35,
    "op": 36,
    # "skip_connections": [37, ...]
}

reversed_attribute = {
    0: {'op': 'dilations', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    2: {'op': 'group', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    3: {'op': 'kernel_shape', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    5: {'op': 'pads', 'len': 4, 'type': 'int', 'range': [0, float('inf')]},
    9: {'op': 'strides', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    11: {'op': 'output_shape', 'len': 4, 'type': 'int', 'range': [0, float('inf')]},
    15: {'op': 'alpha', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    16: {'op': 'axis', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    17: {'op': 'output_shape', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    19: {'op': 'beta', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    20: {'op': 'transB', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    21: {'op': 'epsilon', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    22: {'op': 'momentum', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    23: {'op': 'mode', 'len': 1, 'type': 'int', 'range': [0, len(pads_to_mods)]},
    24: {'op': 'pads', 'len': 8, 'type': 'int', 'range': [0, float('inf')]},
    32: {'op': 'value', 'len': 1, 'type': 'float', 'range': [0.0, float('inf')]},
    33: {'op': 'axes', 'len': 2, 'type': 'int', 'range': [0, float('inf')]},
    35: {'op': 'keepdims', 'len': 1, 'type': 'int', 'range': [0, float('inf')]},
    36: {'op': 'op', 'len': 1, 'type': 'int', 'range': [0, len(node_to_ops)]},
    37: {'len': 1, 'type': 'int', 'range': [0, NODE_EMBEDDING_DIMENSION - 37]},
    38: {'type': 'int', 'range': [0, float('inf')]},
}

# autoencoder = Autoencoder()
# autoencoder.load_state_dict(torch.load("models/autoencoder.pth"))


class NeuralNetworkGraph(nx.DiGraph):
    """Parse graph from network"""

    def __init__(self, model, test_batch):
        """Initialize structure with embedding for each node from `model` and graph from `HiddenLayer`"""
        super().__init__()
        hl_graph = hl.build_graph(model, test_batch, transforms=None)
        self.__colors = {}
        self.__input_shapes = {}
        self.__id_to_node = {}
        self.embedding = []
        self.__parse_graph(hl_graph)

    @classmethod
    def get_graph(cls, embedding, is_naive=False):
        """Create graph from embedding and return it. Get embedding type of list"""
        graph = cls.__new__(cls)
        super(NeuralNetworkGraph, graph).__init__()
        # TODO:
        #  graph.embedding = embedding if is_naive else autoencoder.decode(
        #     torch.tensor(NeuralNetworkGraph.replace_none_in_embedding(embedding, is_need_replace=False))).tolist()
        graph.embedding = embedding
        # graph.embedding = cls.__fix_attributes(graph.embedding)
        graph.__create_graph()
        return graph

    @staticmethod
    def __fix_attributes(embedding):
        for e in range(len(embedding)):
            for pos, attr in reversed_attribute.items():
                if 'len' not in attr:
                    n = NODE_EMBEDDING_DIMENSION - pos
                else:
                    n = attr['len']
                for i in range(n):
                    if not embedding[e][pos + i]:
                        continue
                    if attr['type'] == 'int':
                        embedding[e][pos + i] = int(round(embedding[e][pos + i]))
                    if attr['type'] == 'float':
                        embedding[e][pos + i] = float(embedding[e][pos + i])
                    if embedding[e][pos + i] < attr['range'][0]:
                        embedding[e][pos + i] = attr['range'][0]
                    if attr['range'][1] <= embedding[e][pos + i]:
                        embedding[e][pos + i] = attr['range'][1]
        return embedding

    def get_naive_embedding(self):
        """Return naive embedding"""
        return self.embedding

    @staticmethod
    def replace_none_in_embedding(embedding, is_need_replace=True):
        for i in range(len(embedding)):
            for j in range(len(embedding[i])):
                if is_need_replace and not embedding[i][j]:
                    embedding[i][j] = NONE_REPLACEMENT
                if not is_need_replace and round(embedding[i][j]) == NONE_REPLACEMENT:
                    embedding[i][j] = None
        return embedding

    def get_embedding(self):
        """Return embedding"""
        # TODO:
        #  return autoencoder.encode(
        #     torch.tensor(NeuralNetworkGraph.replace_none_in_embedding(self.embedding.copy()))).tolist()

    def __create_graph(self):
        """Create `networkx.DiGraph` graph from embedding"""
        counter = 0
        for embedding in self.embedding:
            """Add node with attributes to graph"""
            params = {}
            for pos in reversed_attribute:
                if 'op' not in reversed_attribute[pos]:
                    continue
                is_set = True
                if reversed_attribute[pos]['len'] > 1:
                    attr = []
                    for i in range(reversed_attribute[pos]['len']):
                        if embedding[pos + i] is None:
                            is_set = False
                            break
                        attr.append(embedding[pos + i])
                else:
                    if embedding[pos] is None:
                        is_set = False
                    else:
                        if pos == 36:  # Attribute 'op'
                            attr = str(list(filter(lambda x: node_to_ops[x] == embedding[pos], node_to_ops))[0])
                        elif pos == 23:  # Attribute 'mode'
                            attr = str(list(filter(lambda x: pads_to_mods[x] == embedding[pos], pads_to_mods))[0])
                        else:
                            attr = embedding[pos]
                if is_set:
                    params[reversed_attribute[pos]['op']] = attr
            self.add_node(counter, **params)

            """Add edge to graph"""
            for i in range(embedding[ATTRIBUTES_POS_COUNT]):
                self.add_edge(counter, embedding[ATTRIBUTES_POS_COUNT + i + 1])
            counter += 1

    def __add_edges(self, graph):
        """Add edges with changed node's names"""
        for edge in graph.edges:
            v = self.__id_to_node[edge[0]]
            u = self.__id_to_node[edge[1]]
            self.__input_shapes[u] = edge[2]
            self.add_edge(v, u)

    def __is_supported(self, v):
        """Check if graph is supported"""
        # TODO: change
        self.__colors[v] = 1
        result = True
        for u in self.adj[v]:
            if self.__colors.get(u, 0) == 0:
                result &= self.__is_supported(u)
            elif self.__colors.get(u, 0) == 1:
                result = False
        self.__colors[v] = 2
        return result

    def __calculate_embedding(self):
        """Calculate embedding for each node"""
        for id in self.nodes:
            node = self.nodes[id]
            embedding = [None] * NODE_EMBEDDING_DIMENSION

            """
            Take output_shape and check it. output_shape might be None or
            size 2 (for linear), size 4 (for convolutional).
            """
            if not node['output_shape'] or node['output_shape'] == []:
                output_shape = self.__input_shapes.get(id)
                if output_shape:
                    node['output_shape'] = output_shape
                    self.nodes[id]['output_shape'] = output_shape
                else:
                    del node['output_shape']

            """
            Set node's parameters to embedding vector in order described in attribute_to_pos dictionary 
            and map string parameters to its' numeric representation.
            """
            for param in node:
                op_name = param
                if isinstance(node[param], list):
                    if param in ops_with_different_dims:
                        op_name += '_' + str(len(node[param]))
                    current_poses = attribute_to_pos[op_name]
                    for i in range(len(node[param])):
                        embedding[current_poses[i]] = node[param][i]
                else:
                    value = node[param]
                    if param == 'op':
                        value = node_to_ops[value]
                    if param == 'mode' and node['op'] == 'Pad':
                        value = pads_to_mods[value]
                    embedding[attribute_to_pos[op_name]] = value

            edge_list = list(self.adj[id])
            if len(edge_list) + ATTRIBUTES_POS_COUNT + 1 <= 1000:
                embedding[ATTRIBUTES_POS_COUNT] = len(edge_list)
                for i in range(0, len(edge_list)):
                    embedding[ATTRIBUTES_POS_COUNT + i + 1] = edge_list[i]
            else:
                print('This graph is not supported!')
            self.embedding.append(embedding)

    def __parse_graph(self, graph):
        """Parse `HiddenLayer` graph and create `networkx.DiGraph` with same node attributes"""
        try:
            counter = 0

            """Renumber nodes and add it to graph"""
            for id in graph.nodes:
                self.__id_to_node[id] = counter
                graph.nodes[id].params['output_shape'] = graph.nodes[id].output_shape
                graph.nodes[id].params['op'] = graph.nodes[id].op
                self.add_node(counter, **graph.nodes[id].params)
                counter += 1

            self.__add_edges(graph)
            is_supported = self.__is_supported(0)

            if is_supported:
                self.__calculate_embedding()
            else:
                print('Graph is not supported. This network is not supported.')
        except KeyError as e:
            print(f'Operation or layer is not supported: {e}.')

    @staticmethod
    def check_equality(graph1, graph2):
        """Check two graphs on equality. Return if they are equal and message"""
        if graph1.edges != graph2.edges:
            return False, 'Edges are not equal'
        if sorted(list(graph1.nodes)) != sorted(list(graph2.nodes)):
            return False, 'Nodes are not equal'
        for node in graph1.nodes:
            if graph1.nodes[node] != graph2.nodes[node]:
                return False, 'Node params are not equal'
        return True, 'Graphs are equal'
