from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.tf_modules.tf_graphs import SequentialGraphMol
from AGCN.models.networks.basic_networks import Network
from AGCN.models.layers import MLP, DenseMol, SGC_LL, GraphGatherMol


class SimpleAGCN(Network):
    """
    A simple example of AGCN
    """
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']

        # assign the number of feature at output of the SGC_LL layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]

        """ Network Architecture - 4 SGC layers, most original AGCN"""
        self.graph_model = SequentialGraphMol(n_features, batch_size, self.max_atom)

        self.graph_model.add(SGC_LL(n_filters_1, n_features, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_3, n_filters_2, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class MLP_AGCN(Network):

    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        MLP_hidden_dim = self.hyper_parameters['MLP_hidden_dim']
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']

        # assign the number of feature at output of the SGC_LL layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]

        """ Network Architecture - 4 SGC layers, most original AGCN"""
        self.graph_model = SequentialGraphMol(n_features, batch_size, self.max_atom)

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

        self.graph_model.add(SGC_LL(n_filters_1, n_filters_1, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_3, n_filters_2, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(SGC_LL(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class LongAGCN(Network):
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
        # n_filters_5 = l_n_filters[4]
        # n_filters_6 = l_n_filters[5]

        """ Network Architecture - 12 SGC layers"""
        self.graph_model = SequentialGraphMol(n_features, batch_size, self.max_atom)
        self.graph_model.add(SGC_LL(n_filters_1, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_1, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_1, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_1, n_filters_1, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_2, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_2, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_2, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters_3, n_filters_2, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_3, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_4, n_filters_4, batch_size, K=K, activation='relu'))

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")
