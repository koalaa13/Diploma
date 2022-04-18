import sys

import networkx as nx
from torch import nn

from embedding.mapping import NetworkMapping


class Converter:
    """Convert input graph to neural network."""

    def __init__(self, graph, filepath='models/my_model.py', model_name="Model"):
        self.graph = graph
        self.operations = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.node_to_layer = {}
        self.node_to_operation = {}
        self.tabulation = '    '
        first_dim_node = 0 if 'output_shape' in self.graph.nodes[0] else next(iter(self.graph.adj[0]))
        self.out_dim = {0: self.graph.nodes[first_dim_node]['output_shape'][1]}
        self.sequences = {}
        self.is_flatten_needed = True
        self.__default_input_shape = None
        self.__init_first_sequence()
        self._graph_seq = nx.DiGraph()
        self._graph_seq.add_node(1, **{'op': self.graph.nodes[0]['op']})
        self.node_to_sequence = {0: 1}
        self._top_sorted_nodes = []
        self._top_sort_visited = set()
        self.__create_layers(0)

        with open(filepath, 'w') as file:
            self.__write_model_init(file, model_name)
            self.__write_layers(file)
            self.__write_forward(file)

    def __init_first_sequence(self):
        # TODO default input_shape hard coded here
        if self.graph.nodes[0]['op'] == 'Conv' or self.graph.nodes[0]['op'] == 'BatchNorm' \
                or self.graph.nodes[0]['op'] == 'MaxPool' or self.graph.nodes[0]['op'] == 'ConvTranspose':
            self.__default_input_shape = '1'
            self.sequences = {1: [NetworkMapping.map_node(self.graph.nodes[0], 'in_shape', self.out_dim[0])]}
        else:
            self.__default_input_shape = '28 * 28'
            self.is_flatten_needed = False
            self.sequences = {
                1: [NetworkMapping.map_node(self.graph.nodes[0], 'in_shape', self.out_dim[0])]
            }

    def __create_layers(self, cur_node):
        edges = self.graph.adj.get(cur_node)
        current_sequence = len(self.sequences)

        for v in edges:
            if v in self.node_to_sequence:
                continue
            node = self.graph.nodes[v]
            # Skip concat and add
            layer = None
            old_dim = self.out_dim[cur_node]
            self.out_dim[v] = node['output_shape'][1] if node['output_shape'] else old_dim
            if node['op'] not in ['Concat', 'Add']:
                if self.graph.nodes[cur_node]['op'] != 'Pad' and node['op'] == 'AveragePool':
                    node['op'] = 'AdaptiveAveragePool'
                layer = NetworkMapping.map_node(node, old_dim, self.out_dim[v])
            if len(edges) > 1 \
                    or len(self.graph.pred[v]) > 1 \
                    or (
                    node['op'] in ['Concat', 'Pad', 'ReduceMean', 'Slice', 'Reshape'] and len(self.graph.pred[v]) <= 1):
                current_sequence = len(self.sequences) + 1
                if current_sequence not in self._graph_seq.nodes:
                    if node['op'] == 'Pad':
                        self._graph_seq.add_node(current_sequence, **{'op': node['op'], 'pads': node['pads']})
                    elif node['op'] == 'ReduceMean':
                        self._graph_seq.add_node(current_sequence, **{'op': node['op'], 'axes': node['axes']})
                    elif node['op'] == 'Slice':
                        self._graph_seq.add_node(current_sequence, **{'op': node['op'], 'node': node})
                    elif node['op'] == 'Reshape':
                        self._graph_seq.add_node(current_sequence, **{'op': node['op'], 'value': node['value']})
                    else:
                        self._graph_seq.add_node(current_sequence, **{'op': node['op']})
            array = self.sequences.get(current_sequence, [])
            if layer:
                if node['op'] in ['Flatten', 'ReduceMean']:
                    self.is_flatten_needed = False
                if node['op'] == 'Linear' and self.is_flatten_needed:
                    array.append(NetworkMapping.map_node({'op': 'Flatten'}))
                if not layer.startswith('#'):
                    array.append(layer)
            self.sequences[current_sequence] = array
            self.node_to_sequence[v] = current_sequence
            self.__create_layers(v)

        current_sequence = self.node_to_sequence[cur_node]
        for v in edges:
            new_sequence = self.node_to_sequence[v]
            if new_sequence == current_sequence:
                continue
            self._graph_seq.add_edge(current_sequence, new_sequence)

    @staticmethod
    def __write_line(file, line, tab=''):
        file.write(tab + line + '\n')

    def __write_layers(self, file):
        for key, value in self.sequences.items():
            if len(value) == 0:
                continue
            Converter.__write_line(file, f'self.seq{key} = nn.Sequential(', self.tabulation * 2)
            for elem in value:
                Converter.__write_line(file, elem + ',', self.tabulation * 3)
            Converter.__write_line(file, ')', self.tabulation * 2)
        Converter.__write_line(file, '')

    def __top_sort(self, cur_node):
        self._top_sort_visited.add(cur_node)
        for v in self._graph_seq.adj[cur_node]:
            if v not in self._top_sort_visited:
                self.__top_sort(v)
        self._top_sorted_nodes.append(cur_node)

    def __write_forward(self, file):
        Converter.__write_line(file, 'def forward(self, x_0):', self.tabulation)

        self.__top_sort(1)
        self._top_sorted_nodes.reverse()

        for v in self._top_sorted_nodes:
            if len(self._graph_seq.pred[v]) > 1:
                if self._graph_seq.nodes[v]['op'] == 'Concat':
                    inputs = []
                    for u in self._graph_seq.pred[v]:
                        inputs.append(f'x_{u}')
                    Converter.__write_line(file, f'x_{v} = torch.cat([{", ".join(map(str, inputs))}], 1)',
                                           self.tabulation * 2)
                    if len(self.sequences[v]) > 0:
                        Converter.__write_line(file,
                                               f'x_{v} = self.seq{v}(x_{v})',
                                               self.tabulation * 2)
                if self._graph_seq.nodes[v]['op'] == 'Add':
                    inputs = []
                    for u in self._graph_seq.pred[v]:
                        inputs.append(f'x_{u}')
                    Converter.__write_line(file, f'x_{v} = {" + ".join(map(str, inputs))}',
                                           self.tabulation * 2)
                    if len(self.sequences[v]) > 0:
                        Converter.__write_line(file,
                                               f'x_{v} = self.seq{v}(x_{v})',
                                               self.tabulation * 2)
            else:
                cur_x_seq = next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))
                if self._graph_seq.nodes[v]['op'] == 'Concat':
                    Converter.__write_line(file,
                                           f'x_{v} = torch.cat([x_{next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))}], 1)',
                                           self.tabulation * 2)
                    cur_x_seq = v
                elif self._graph_seq.nodes[v]['op'] == 'Pad':
                    cur_pads = self._graph_seq.nodes[v]["pads"]
                    new_pads = []
                    for i in range(len(cur_pads) // 2):
                        new_pads.append(cur_pads[len(cur_pads) - i - 1 - len(cur_pads) // 2])
                        new_pads.append(cur_pads[len(cur_pads) - i - 1])
                    prev_seq = next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))
                    if new_pads != [0] * len(new_pads):
                        Converter.__write_line(file,
                                               f'x_{v} = torch.nn.functional.pad(x_{prev_seq}, {new_pads})',
                                               self.tabulation * 2)
                        cur_x_seq = v
                elif self._graph_seq.nodes[v]['op'] == 'ReduceMean':
                    prev_seq = next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))
                    Converter.__write_line(file,
                                           f'x_{v} = x_{prev_seq}.mean({self._graph_seq.nodes[v]["axes"]})',
                                           self.tabulation * 2)
                    cur_x_seq = v
                elif self._graph_seq.nodes[v]['op'] == 'Slice':
                    prev_seq = next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))
                    cur_node = self._graph_seq.nodes[v]["node"]
                    tensor_slice = [':', ':', ':', ':'][:len(cur_node["output_shape"])]
                    for i in range(len(cur_node["axes"])):
                        start = str(cur_node["starts"][i]) if cur_node["starts"][i] != sys.maxsize else ''
                        end = str(cur_node["ends"][i]) if cur_node["ends"][i] != sys.maxsize else ''
                        tensor_slice[cur_node["axes"][i]] = start + ':' + end
                        if len(cur_node.get("steps", [])) > 0:
                            step = str(cur_node["steps"][i]) if cur_node["steps"][i] != sys.maxsize else ''
                            tensor_slice[cur_node["axes"][i]] += ':' + step
                    Converter.__write_line(file,
                                           f'x_{v} = x_{prev_seq}[{", ".join(tensor_slice)}]',
                                           self.tabulation * 2)
                    cur_x_seq = v
                elif self._graph_seq.nodes[v]['op'] == 'Reshape':
                    prev_seq = next(iter(self._graph_seq.pred[v] if self._graph_seq.pred.get(v) else {0}))
                    Converter.__write_line(file,
                                           f'x_{v} = x_{prev_seq}.view({", ".join(list(map(str, self._graph_seq.nodes[v]["value"])))})',
                                           self.tabulation * 2)
                    cur_x_seq = v

                if self.sequences[v]:
                    Converter.__write_line(file,
                                           f'x_{v} = self.seq{v}(x_{cur_x_seq})',
                                           self.tabulation * 2)

        Converter.__write_line(file, f'return x_{self._top_sorted_nodes[len(self._top_sorted_nodes) - 1]}',
                               self.tabulation * 2)

    def __write_model_init(self, file, model_name):
        Converter.__write_line(file, 'import torch')
        Converter.__write_line(file, 'from torch import nn\n\n')
        Converter.__write_line(file, f'class {model_name}(nn.Module):\n')
        Converter.__write_line(file,
                               f'def __init__(self, in_shape={self.__default_input_shape}):',
                               self.tabulation)
        Converter.__write_line(file, f'super({model_name}, self).__init__()', self.tabulation * 2)
