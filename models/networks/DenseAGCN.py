from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.networks.basic_networks import SimpleAGCN
from AGCN.models.tf_modules.tf_graphs import DenseConnectedGraph, DenseConnectedGraphResLap
from AGCN.models.layers import DenseMol, SGC_LL, SGC_LL_Reslap, GraphGatherMol, GraphPoolMol, DenseBlockEnd
# from AGCN.models.tf_modules.multitask_classifier import MultitaskGraphClassifier


class DenseAGCN(SimpleAGCN):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
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
        self.graph_model = DenseConnectedGraph(2, n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL(
            n_filters_1,
            n_features,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(SGC_LL(
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
        """ block end """
        # self.graph_model.add(SGC_LL(
        #     n_filters_2,
        #     n_filters_2,
        #     batch_size,
        #     K=K,
        #     activation='relu',
        #     save_output=True)
        # )
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(
            n_filters_3,
            n_filters_2,
            batch_size,
            K=K,
            activation='relu',
            save_output=True)
        )
        self.graph_model.add(SGC_LL(
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

        # self.graph_model.add(SGC_LL(n_filters_4, n_filters_4, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class DenseAGCNResLap(SimpleAGCN):
    """
    This DenseNet adds all residual activations from both in-block/out-block layers, and it added
    graph Laplacian residuals from most recent convolution layer
    """
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
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


class LongDenseAGCN(DenseAGCN):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']

        # assign the number of feature at output of the layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]
        n_filters_5 = l_n_filters[2]
        n_filters_6 = l_n_filters[3]

        """ Residual Network Architecture - 3 dense net blocks, 9 SGC layers"""
        self.graph_model = DenseConnectedGraph(n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL(n_filters_1, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(
            1,
            'relu',
            self.max_atom,
            batch_size))
        """ block end """
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_2, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_3, n_filters_2, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(
            2,
            'relu',
            self.max_atom,
            batch_size))

        self.graph_model.add(SGC_LL(n_filters_4, n_filters_4, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_5, n_filters_4, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_6, n_filters_5, batch_size, K=K, activation='relu'))
        self.graph_model.add(DenseBlockEnd(
            3,
            'relu',
            self.max_atom,
            batch_size))

        self.graph_model.add(SGC_LL(n_filters_6, n_filters_6, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_6, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")
