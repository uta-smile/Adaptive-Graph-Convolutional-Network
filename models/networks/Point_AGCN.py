from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.networks.basic_AGCN import Network
from AGCN.models.tf_modules.tf_graphs import DenseConnectedGraph, DenseConnectedGraphResLap
from AGCN.models.layers import MLP, DenseMol, SGC_LL, SGC_LL_Reslap, GraphGatherMol, GraphPoolMol, DenseBlockEnd


class Point_AGCNResLap(Network):
    """
    This DenseNet adds all residual activations from both in-block/out-block layers, and it added
    graph Laplacian residuals from most recent convolution layer
    """
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.hyper_parameters['raw_feature_n']
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']

        # assign the number of feature at output of the layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]

        """ Residual Network Architecture - 2 dense net blocks, 6 SGC layers"""
        self.graph_model = DenseConnectedGraphResLap(2, n_features, batch_size, self.max_atom)
        """ block start """

        self.graph_model.add(SGC_LL_Reslap(
            n_filters_1,
            n_features,
            batch_size,
            K=K,
            activation='relu',
            save_lap=True,
            save_output=True)
        )
        self.graph_model.add(SGC_LL_Reslap(
            n_filters_2,
            n_filters_1,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(DenseBlockEnd(
            0,
            [n_filters_1, n_filters_2],
            n_filters_2,
            'relu',
            max_atom=self.max_atom,
            batch_size=batch_size))

        self.graph_model.add(SGC_LL_Reslap(
            n_filters_3,
            n_filters_2,
            batch_size,
            K=K,
            activation='relu',
            save_lap=True,
            save_output=True)
        )
        self.graph_model.add(SGC_LL_Reslap(
            n_filters_4,
            n_filters_3,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(DenseBlockEnd(
            1,
            [n_filters_3, n_filters_4],
            n_filters_4,
            'relu',
            max_atom=self.max_atom,
            batch_size=batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class Point_MLPDenseAGCNResLap(Network):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.hyper_parameters['raw_feature_n']
        MLP_hidden_dim = self.hyper_parameters['MLP_hidden_dim']
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']

        # assign the number of feature at output of the layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]

        """ Residual Network Architecture - 2 dense net blocks, 6 SGC layers"""
        self.graph_model = DenseConnectedGraphResLap(2, n_features, batch_size, self.max_atom)

        self.graph_model.add(MLP(
            n_filters_1,
            MLP_hidden_dim,
            n_features,
            batch_size,
            init='glorot_uniform',
            activation="relu",
            bias=True,
            max_atom=self.max_atom,
        ))

        """ block start """
        self.graph_model.add(SGC_LL_Reslap(
            n_filters_1,
            n_filters_1,
            batch_size,
            K=K,
            activation='relu',
            save_lap=True,
            save_output=True)
        )
        self.graph_model.add(SGC_LL_Reslap(
            n_filters_2,
            n_filters_1,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(DenseBlockEnd(
            0,
            [n_filters_1, n_filters_2],
            n_filters_2,
            'relu',
            max_atom=self.max_atom,
            batch_size=batch_size))

        self.graph_model.add(SGC_LL_Reslap(
            n_filters_3,
            n_filters_2,
            batch_size,
            K=K,
            activation='relu',
            save_lap=True,
            save_output=True)
        )
        self.graph_model.add(SGC_LL_Reslap(
            n_filters_4,
            n_filters_3,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(DenseBlockEnd(
            1,
            [n_filters_3, n_filters_4],
            n_filters_4,
            'relu',
            max_atom=self.max_atom,
            batch_size=batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")
