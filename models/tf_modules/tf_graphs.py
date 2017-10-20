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
        self.L_set = []     # stored tensor for residual Laplacian matrix for each sample in batch

    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['GraphGatherMol', 'GraphPoolMol', 'SGC_LL']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:
                    self.output, res_L = layer(self.output + self.graph_topology.get_topology_placeholders())
                    self.L_set.extend(res_L)
                else:
                    self.output = layer(self.output + self.graph_topology.get_topology_placeholders())
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)

    def return_L_set(self):
        return self.L_set   # return the Laplacian of each output of  gcn convolution layer

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

    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['GraphGatherMol', 'GraphPoolMol', 'SGC_LL', 'BlockEnd']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:
                    self.output, res_L = layer(self.output + self.graph_topology.get_topology_placeholders())
                    self.L_set.extend(res_L)
                    if 'SGC_LL' not in map(lambda l: type(l).__name__, self.layers):
                        # first SGC_LL layer
                        self.block_outputs.append(self.output)
                elif type(layer).__name__ in ['BlockEnd']:  # BlockEnd layer add saved last block output
                    self.output = layer(
                        self.output + [self.graph_topology.get_dataslice_placeholders()] + self.block_outputs[-1]
                    )
                    self.block_outputs.append(self.output)
                else:
                    self.output = layer(self.output + self.graph_topology.get_topology_placeholders())
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)


class DenseConnectedGraph(ResidualGraphMol):
    """
    tf graph for densely connected network
    """
    def add(self, layer):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            if type(layer).__name__ in ['GraphGatherMol', 'GraphPoolMol', 'SGC_LL', 'DenseBlockEnd']:
                if len(self.layers) > 0 and hasattr(self.layers[-1], "__name__"):
                    assert self.layers[-1].__name__ != "GraphGatherMol", \
                        'Cannot use GraphConv or GraphGather layers after a GraphGather'
                if type(layer).__name__ in ['SGC_LL']:
                    self.output, res_L = layer(self.output + self.graph_topology.get_topology_placeholders())
                    self.L_set.extend(res_L)
                    self.block_outputs += self.output   # self.block_outputs is still a list
                elif type(layer).__name__ in ['DenseBlockEnd']:  # BlockEnd layer add saved last block output
                    self.output = layer(
                        self.output + [self.graph_topology.get_dataslice_placeholders()] + self.block_outputs
                    )
                    # in next Dense Block, re-stack the convolution activations
                    self.block_outputs = []
                else:
                    self.output = layer(self.output + self.graph_topology.get_topology_placeholders())
            else:
                self.output = layer(self.output)
            # Add layer to the layer list
            self.layers.append(layer)
