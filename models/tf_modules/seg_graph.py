from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
from AGCN.models.tf_modules.graph_topology import GraphTopologyMol


class SegmentationGraph(object):
    """
    Tensorflow graph for Segmentation network
    1. classification network
    2. segmentation network
    """
    def __init__(self,
                 n_feat,
                 batch_size=50,
                 num_point=128):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # given different GraphTopologyMol, do placeholder mol wise
            self.graph_topology = GraphTopologyMol(n_feat, batch_size, num_point)
            """this output, at the beginning, is the input placeholders, data, Laplacian and their shape slice """
            self.output = self.graph_topology.get_atom_features_placeholder()
        # Keep track of the layers
        self.backbone_layers = []
        self.classifier_layer = []
        self.segmentation_layer = []
        self.num_point = num_point
        self.batch_size = batch_size
        self.classifier_output = None
        self.segmentation_output = None

    def add(self, layer, classifer=False, segmentation=False):
        """Adds a new layer to model."""
        with self.graph.as_default():
            # For graphical layers, add connectivity placeholders
            assert type(layer).__name__ in ['MLP', 'GraphGatherMol', 'GraphPoolMol', 'SGC_LL',
                                            'DenseMol', 'FCL', 'Merge']
            if classifer:
                assert len(self.backbone_layers) > 0
                if type(layer).__name__ in ['FCL']:
                    input = dict()
                    input['node_features'] = self.classifier_output  # node features
                    self.classifier_output = layer(input)
                else:
                    # GraphGatherMol and GraphPoolMol, and DenseMOl
                    input = dict()
                    input['node_features'] = self.classifier_output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.classifier_output = layer(input)

                self.classifier_layer.append(layer)

            elif segmentation:
                assert len(self.backbone_layers) > 0
                if type(layer).__name__ in ['SGC_LL']:
                    input = dict()
                    input['node_features'] = self.segmentation_output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()

                    self.segmentation_output, res_L, res_W = layer(input)
                elif type(layer).__name__ in ['Merge']:
                    self.segmentation_output = layer(self.segmentation_output)
                else:
                    # GraphGatherMol and GraphPoolMol,and DenseMOl,
                    input = dict()
                    input['node_features'] = self.segmentation_output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.segmentation_output = layer(input)

                self.segmentation_layer.append(layer)

            else:
                """ add layer to backbone part, shared by both segmentation and classification"""
                if type(layer).__name__ in ['SGC_LL']:
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output, res_L, res_W = layer(input)

                elif type(layer).__name__ in ['MLP']:
                    input = dict()
                    input['node_features'] = self.output
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    self.output = layer(input)

                else:
                    # GraphGatherMol and GraphPoolMol, and DenseMOl
                    input = dict()
                    input['node_features'] = self.output  # node features
                    input['data_slice'] = self.graph_topology.get_dataslice_placeholders()
                    input['original_laplacian'] = self.graph_topology.get_laplacians_placeholder()
                    input['lap_slice'] = self.graph_topology.get_lapslice_placeholders()
                    self.output = layer(input)

                self.classifier_output = self.output
                self.segmentation_output = self.output

                # Add layer to the backbone (shared) layer list
                self.backbone_layers.append(layer)

    def get_graph_topology(self):
        return self.graph_topology

    def backbone_outputs(self):
        return self.output

    def classification_outputs(self):
        return self.classifier_output

    def segmentation_outputs(self):
        return self.segmentation_output

    def return_inputs(self):
        return self.graph_topology.get_input_placeholders()

