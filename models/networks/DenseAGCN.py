from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.networks.basic_networks import SimpleAGCN
from AGCN.models.tf_modules.tf_graphs import DenseConnectedGraph
from AGCN.models.layers import DenseMol, SGC_LL, GraphGatherMol, GraphPoolMol, DenseBlockEnd
from AGCN.models.tf_modules.multitask_classifier import MultitaskGraphClassifier


class DenseAGCN(SimpleAGCN):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        n_filters = self.hyper_parameters['n_filters']  # SGL_LL output dimensions
        final_feature_n = self.hyper_parameters['final_feature_n']

        """ Residual Network Architecture - 2 dense net blocks, 6 SGC layers"""
        self.graph_model = DenseConnectedGraph(n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL(n_filters, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(1, 'relu', self.max_atom, batch_size))
        """ block end """
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(2, 'relu', self.max_atom, batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class LongDenseAGCN(DenseAGCN):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        n_filters = self.hyper_parameters['n_filters']  # SGL_LL output dimensions
        final_feature_n = self.hyper_parameters['final_feature_n']

        """ Residual Network Architecture - 4 dense net blocks, 12 SGC layers"""
        self.graph_model = DenseConnectedGraph(n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL(n_filters, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(1, 'relu', self.max_atom, batch_size))
        """ block end """
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(2, 'relu', self.max_atom, batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(3, 'relu', self.max_atom, batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(4, 'relu', self.max_atom, batch_size))

        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")
