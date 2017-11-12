"""
Convenience classes for assembling graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
from AGCN.models.tf_modules.graph_topology import GraphTopologyMol


class SequentialGraphMol(object):
    """
    inherit most of SequentialGraph expect use GraphTopology_mol instead of
    GraphTopology
    """

    def __init__(self, n_feat, batch_size=50, max_atom=128):
        """
        Parameters
        ----------
        n_feat: int
          Number of features per node.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            # given different GraphTopologyMol, do placeholder mol wise
            self.graph_topology = GraphTopologyMol(n_feat, batch_size, max_atom)
            self.output = self.graph_topology.get_atom_features_placeholder()
        # Keep track of the layers
        self.layers = []
        self.max_atom = max_atom
        self.batch_size = batch_size
        self.res_L_set = []     # stored tensor for residual Laplacian matrix for each sample in batch
        self.res_W_set = []

    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol', 'SGC_LL', 'DenseMol']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()

                    self.output, res_L, res_W = layer(input)
                    self.res_L_set.extend(res_L)
                    self.res_W_set.extend(res_W)

                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)
                else:
                    # GraphGatherMol and GraphPoolMol
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)

            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)

    def get_resL_set(self):
        return self.res_L_set   # return the Laplacian of each output of  gcn convolution layer

    def get_resW_set(self):
        return self.res_W_set

    def get_graph_topology(self):
        return self.graph_topology

    def get_num_output_features(self):
        """Gets the output shape of the featurization layers of the network"""
        return self.layers[-1].output_shape[1]

    def return_outputs(self):
        return self.output

    def return_inputs(self):
        return self.graph_topology.get_input_placeholders()

    def get_layer(self, layer_id):
        return self.layers[layer_id]


class ResidualGraphMol(SequentialGraphMol):

    def __init__(self, *args, **kwargs):
        super(ResidualGraphMol, self).__init__(*args, **kwargs)
        self.block_outputs = []  # saved the output tensors of each block

    """
    tf graph for residual network on graphs
    """
    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol', 'SGC_LL', 'BlockEnd', 'DenseMol']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:

                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()

                    self.output, res_L, res_W = layer(input)
                    self.res_L_set.extend(res_L)
                    self.res_W_set.extend(res_W)
                    if 'SGC_LL' not in map(lambda l: type(l).__name__, self.layers):
                        # first SGC_LL layer, use it initial block residual
                        self.block_outputs.append(self.output)

                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)

                elif type(layer).__name__ in ['BlockEnd']:  # BlockEnd layer add saved last block output
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['block_outputs'] = self.block_outputs[-1]

                    self.output = layer(input)
                    self.block_outputs.append(self.output)

                else:
                    # GraphGatherMol and GraphPoolMol
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)


class ResidualGraphMolResLap(SequentialGraphMol):

    def __init__(self, *args, **kwargs):
        super(ResidualGraphMolResLap, self).__init__(*args, **kwargs)
        self.block_outputs = []  # saved the output tensors of each block
        self.stack_laplacians = []

    """
    tf graph for residual network on graphs + residual graph Laplacian added to later layers
    """
    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol', 'SGC_LL_Reslap',
                                        'BlockEnd', 'DenseMol']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL_Reslap']:
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    input['res_lap'] = self.stack_laplacians[-self.batch_size:]
                    self.output, res_L, res_W, L = layer(input)

                    self.res_L_set.extend(res_L)
                    self.res_W_set.extend(res_W)

                    if 'SGC_LL_Reslap' not in map(lambda l: type(l).__name__, self.layers):
                        # first SGC_LL layer
                        self.block_outputs.append(self.output)
                    if layer.save_lap:  # save the laplacian matrix
                        self.stack_laplacians.extend(L)     # extend as a list, whose length is batch size
                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)

                elif type(layer).__name__ in ['BlockEnd']:  # BlockEnd layer add saved last block output
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['block_outputs'] = self.block_outputs[-1]

                    self.output = layer(input)
                    self.block_outputs.append(self.output)
                else:
                    # GraphGatherMol and GraphPoolMol
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)

    def get_laplacian(self):
        return self.stack_laplacians


class DenseConnectedGraph(SequentialGraphMol):
    """
    This graph network, is compatible with SGC_LL, no residual graph stacked
    """
    def __init__(self, n_blocks, *args, **kwargs):
        super(DenseConnectedGraph, self).__init__(*args, **kwargs)
        # save preceding activations within the same dense block
        self.inblock_activations = {}
        self.inblock_activations_dim = {}   # save the activation dimension of preceding layers
        # save preceding blocks' outputs before this block
        self.block_outputs = []
        self.block_outputs_dim = []     # save the output dimension of preceding blocks
        self.current_block_id = 0   # used in construct blocks

        for b_id in range(n_blocks):
            # one block may contain many layers , use list
            self.inblock_activations[b_id] = []
            self.inblock_activations_dim[b_id] = []

    """
    tf graph for densely connected network on graphs
    """
    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol', 'SGC_LL',
                                        'DenseBlockEnd', 'DenseMol']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()

                    self.output, res_L, res_W = layer(input)
                    self.res_L_set.extend(res_L)
                    self.res_W_set.extend(res_W)

                    """add activation to dict"""
                    if layer.save_output:
                        self.inblock_activations[self.current_block_id] += self.output
                        self.inblock_activations_dim[self.current_block_id] += [layer.nb_filter]

                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)

                elif type(layer).__name__ in ['DenseBlockEnd']:  # BlockEnd layer add saved last block output

                    assert len(self.inblock_activations[layer.block_id]) > 0
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['inblock_activations'] = self.inblock_activations[layer.block_id]
                    input['inblock_activations_dim'] = self.inblock_activations_dim[layer.block_id]
                    input['block_outputs'] = self.block_outputs
                    input['block_outputs_dim'] = self.block_outputs_dim

                    self.output = layer(input)

                    self.block_outputs += self.output
                    self.block_outputs_dim += [layer.output_n_features]
                    self.current_block_id += 1

                else:
                    # GraphGatherMol and GraphPoolMol
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)


class DenseConnectedGraphResLap(SequentialGraphMol):
    """
       This graph network, is compatible with SGC_LL, no residual graph stacked
       """
    def __init__(self, n_blocks, *args, **kwargs):
        super(DenseConnectedGraphResLap, self).__init__(*args, **kwargs)
        # save preceding activations within the same dense block
        self.inblock_activations = {}
        self.inblock_activations_dim = {}  # save the activation dimension of preceding layers
        # save preceding blocks' outputs before this block
        self.block_outputs = []
        self.block_outputs_dim = []  # save the output dimension of preceding blocks
        self.current_block_id = 0  # used in construct blocks

        for b_id in range(n_blocks):
            # one block may contain many layers , use list
            self.inblock_activations[b_id] = []
            self.inblock_activations_dim[b_id] = []

        self.stack_laplacians = []

    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol',
                                        'SGC_LL_Reslap', 'DenseBlockEnd', 'DenseMol']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL_Reslap']:
                    input = dict()
                    input['node_features'] = self.output   # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    input['res_lap'] = self.stack_laplacians[-self.batch_size:]

                    # self.output, res_L, res_W, L = layer(
                    #     self.output + self.graph_topology.get_topology_placeholders() +
                    #     self.stack_laplacians[-self.batch_size:]
                    # )
                    self.output, res_L, res_W, L = layer(input)
                    self.res_L_set.extend(res_L)
                    self.res_W_set.extend(res_W)

                    """add activation to dict"""
                    if layer.save_output:
                        self.inblock_activations[self.current_block_id] += self.output
                        self.inblock_activations_dim[self.current_block_id] += [layer.nb_filter]
                    if layer.save_lap:  # save the laplacian matrix
                        self.stack_laplacians.extend(L)  # extend as a list, whose length is batch size

                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)

                elif type(layer).__name__ in ['DenseBlockEnd']:  # BlockEnd layer add saved last block output

                    assert len(self.inblock_activations[layer.block_id]) > 0
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['inblock_activations'] = self.inblock_activations[layer.block_id]
                    input['inblock_activations_dim'] = self.inblock_activations_dim[layer.block_id]
                    input['block_outputs'] = self.block_outputs
                    input['block_outputs_dim'] = self.block_outputs_dim

                    self.output = layer(input)
                    self.block_outputs += self.output
                    self.block_outputs_dim += [layer.output_n_features]
                    self.current_block_id += 1

                else:
                    # GraphGatherMol and GraphPoolMol
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)

    def get_laplacian(self):
        return self.stack_laplacians

