from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.tf_modules.seg_graph import SegmentationGraph
from AGCN.models.layers import MLP, DenseMol, SGC_LL, FCL, GraphGatherMol, GraphPoolMol, Merge
from AGCN.models.tf_modules import PartSegmentation


class SegAGCN(object):
    def __init__(self,
                 train_data,
                 test_data,
                 max_atom,
                 hyper_parameters,
                 ):

        self.data = {'train': train_data, 'testing': test_data}
        self.hyper_parameters = hyper_parameters

        self.max_atom = max_atom
        self.seed = self.hyper_parameters['seed']
        self.graph_model = None  # network architecture
        """
         construct network
         """
        self.construct_network()

        self.classifier = None  # network model for training and evaluation
        self.tasks_type = self.hyper_parameters['task_type']  # regression / classification / segmentation

        """ Define Classifier """
        if self.tasks_type == 'part_segmentation':
            self.classifier = PartSegmentation(
                self.graph_model,
                logdir=self.hyper_parameters['logdir'],
                n_classes=self.hyper_parameters['n_classes'],
                batch_size=self.hyper_parameters['batch_size'],
                num_point=self.max_atom,
                learning_rate=self.hyper_parameters['learning_rate'],
                beta1=self.hyper_parameters['optimizer_beta1'],
                beta2=self.hyper_parameters['optimizer_beta2'],
                n_feature=self.hyper_parameters['final_feature_n']
            )
        else:
            ValueError("task type %s is not defined! " % self.tasks_type)

        self.model_name = self.hyper_parameters['model_name']
        self.data_name = self.hyper_parameters['data_name']

    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.hyper_parameters['raw_feature_num']
        MLP_hidden_dim = self.hyper_parameters['MLP_hidden_dim']
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        final_feature_n = self.hyper_parameters['final_feature_n']
        l_n_filters = self.hyper_parameters['l_n_filters']
        n_classes = self.hyper_parameters['n_classes']
        part_num = self.hyper_parameters['part_num']
        point_num = self.max_atom

        # assign the number of feature at output of the SGC_LL layer
        n_filters_1 = l_n_filters[0]
        n_filters_2 = l_n_filters[1]
        n_filters_3 = l_n_filters[2]
        n_filters_4 = l_n_filters[3]

        """ Network Architecture - 4 SGC layers, most original AGCN"""
        self.graph_model = SegmentationGraph(n_features, batch_size, self.max_atom)

        """backbone network"""
        self.graph_model.add(MLP(
            64,
            MLP_hidden_dim,
            n_features,
            batch_size,
            init='glorot_uniform',
            activation="relu",
            bias=True,
            max_atom=self.max_atom,
        ))

        self.graph_model.add(SGC_LL(128, 64, batch_size, K=K, activation='relu'))
        # self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(256, 128, batch_size, K=K, activation='relu'))
        self.graph_model.add(GraphPoolMol(batch_size))

        """classification network"""
        self.graph_model.add(
            DenseMol(128, 256, activation='relu'),
            classifer=True,
        )
        # output layer for classification
        self.graph_model.add(
            GraphGatherMol(batch_size, activation="tanh"),
            classifer=True,
        )
        self.graph_model.add(
            FCL(batch_size, output_dim=64, input_dim=128),
            classifer=True,
        )
        self.graph_model.add(
            FCL(batch_size, output_dim=n_classes, input_dim=64),
            classifer=True,
        )

        """segmentation network"""
        self.graph_model.add(
            SGC_LL(128, 256, batch_size, K=K, activation='relu'),
            segmentation=True,
        )
        self.graph_model.add(
            SGC_LL(part_num, 128, batch_size, K=K, activation='relu'),
            segmentation=True,
        )
        "create output tensor [batch_size, point_num, part_num]"
        self.graph_model.add(
            Merge(batch_size, point_num=point_num, part_num=part_num),
            segmentation=True,
        )

        print("Network Constructed Successfully! \n")

    def train(self):
        assert self.graph_model is not None
        assert self.classifier is not None  # test if network is built
        print("Start Training ...... \n\n")
        n_epoch = self.hyper_parameters['n_epoch']  # number of epoch for training

        self.classifier.fit(
            self.data['train'],
            self.data['testing'],
            nb_epoch=n_epoch,)

