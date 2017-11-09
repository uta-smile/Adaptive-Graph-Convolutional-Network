from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.networks.basic_AGCN import Network
from AGCN.models.tf_modules.tf_graphs import ResidualGraphMol, ResidualGraphMolResLap
from AGCN.models.layers import MLP, DenseMol, SGC_LL, GraphGatherMol, SGC_LL_Reslap, BlockEnd


class ResAGCN(Network):
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

        """ Residual Network Architecture """
        self.graph_model = ResidualGraphMol(n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL(n_filters_1, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(1,
                                      n_filters_1,
                                      n_filters_2,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )
        """ block end """
        self.graph_model.add(SGC_LL(n_filters_3, n_filters_2, batch_size, K=K, activation='relu'))
        self.graph_model.add(SGC_LL(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(2,
                                      n_filters_3,
                                      n_filters_4,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )

        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class ResAGCNResLap(Network):
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

        """ Residual Network Architecture """
        self.graph_model = ResidualGraphMolResLap(n_features, batch_size, self.max_atom)
        """ block start """
        self.graph_model.add(SGC_LL_Reslap(n_filters_1, n_features, batch_size, K=K, activation='relu', save_lap=True))
        self.graph_model.add(SGC_LL_Reslap(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(1,
                                      n_filters_1,
                                      n_filters_2,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )
        """ block end """
        self.graph_model.add(SGC_LL_Reslap(n_filters_3, n_filters_2, batch_size, K=K, activation='relu', save_lap=True))
        self.graph_model.add(SGC_LL_Reslap(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(2,
                                      n_filters_3,
                                      n_filters_4,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )
        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


class MLP_ResAGCNResLap(Network):
    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
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
        # n_filters_5 = l_n_filters[4]
        # n_filters_6 = l_n_filters[5]

        """ Residual Network Architecture """
        self.graph_model = ResidualGraphMolResLap(n_features, batch_size, self.max_atom)
        """ block start """
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
        self.graph_model.add(SGC_LL_Reslap(n_filters_1, n_filters_1, batch_size, K=K, activation='relu', save_lap=True))
        self.graph_model.add(SGC_LL_Reslap(n_filters_2, n_filters_1, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(1,
                                      n_filters_1,
                                      n_filters_2,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )
        """ block end """
        self.graph_model.add(SGC_LL_Reslap(n_filters_3, n_filters_2, batch_size, K=K, activation='relu', save_lap=True))
        self.graph_model.add(SGC_LL_Reslap(n_filters_4, n_filters_3, batch_size, K=K, activation='relu'))
        self.graph_model.add(BlockEnd(2,
                                      n_filters_3,
                                      n_filters_4,
                                      'relu',
                                      self.max_atom,
                                      batch_size)
                             )
        self.graph_model.add(DenseMol(final_feature_n, n_filters_4, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        print("Network Constructed Successfully! \n")


# class LongResAGCN(ResAGCN):
#     def construct_network(self):
#         tf.set_random_seed(self.seed)
#
#         n_features = self.data['train'].get_raw_feature_n()
#         batch_size = self.hyper_parameters['batch_size']
#         K = self.hyper_parameters['max_hop_K']
#         n_filters = self.hyper_parameters['n_filters']  # SGL_LL output dimensions
#         final_feature_n = self.hyper_parameters['final_feature_n']
#         learning_rate = self.hyper_parameters['learning_rate']
#         beta1 = self.hyper_parameters['optimizer_beta1']
#         beta2 = self.hyper_parameters['optimizer_beta2']
#         optimizer_type = self.hyper_parameters['optimizer_type']
#
#         """ Residual Network Architecture - 6 residual blocks 12 SGC layers"""
#         self.graph_model = ResidualGraphMol(n_features, batch_size, self.max_atom)
#
#         self.graph_model.add(SGC_LL(n_filters, n_features, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(SGC_LL(n_filters, n_features, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
#         self.graph_model.add(BlockEnd(self.max_atom, batch_size))
#
#         self.graph_model.add(DenseMol(final_feature_n, n_filters, activation='relu'))
#         self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))
#
#         """ Classifier """
#         self.classifier = MultitaskGraphClassifier(
#             self.graph_model,
#             len(self.tasks),
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             optimizer_type=optimizer_type,
#             beta1=beta1,
#             beta2=beta2,
#             n_feature=final_feature_n
#         )
#         print("Network Constructed Successfully! \n")