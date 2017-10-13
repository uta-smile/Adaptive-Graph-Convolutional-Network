from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
from AGCN.models.tf_modules.tf_graphs import SequentialGraphMol
from AGCN.models.layers import  DenseMol, SGC_LL, GraphGatherMol, GraphPoolMol
from AGCN.models.tf_modules.multitask_classifier import MultitaskGraphClassifier


class Network(object):
    def __init__(self,
                 train_data,
                 valid_data,
                 test_data,
                 max_atom,
                 tasks,
                 hyper_parameters,
                 transformers,
                 metrics,
                 seed=123,
                 ):

        self.data = {'train': train_data, 'validation': valid_data, 'testing': test_data}
        self.max_atom = max_atom
        self.tasks = tasks
        self.metrics = metrics
        self.transformers = transformers
        self.seed = seed

        assert hyper_parameters is not None
        self.hyper_parameters = hyper_parameters

        self.graph_model = None     # network architecture
        self.classifier = None   # network model for training and evaluation
        self.model_name = ""
        """
        construct network
        """
        self.construct_network()

        self.outputs = {}

    def construct_network(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def final_evaluate(self):
        raise NotImplementedError


class SimpleAGCN(Network):
    """
    A simple example of AGCN
    """
    # def __init__(self,
    #              **kwargs):
    #     super(SimpleAGCN, self).__init__(**kwargs)
    #     self.model_name = 'SimpleAGCN'

    def construct_network(self):
        tf.set_random_seed(self.seed)

        n_features = self.data['train'].get_raw_feature_n()
        batch_size = self.hyper_parameters['batch_size']
        K = self.hyper_parameters['max_hop_K']
        n_filters = self.hyper_parameters['n_filters']  # SGL_LL output dimensions
        final_feature_n = self.hyper_parameters['final_feature_n']
        learning_rate = self.hyper_parameters['learning_rate']
        beta1 = self.hyper_parameters['optimizer_beta1']
        beta2 = self.hyper_parameters['optimizer_beta2']
        optimizer_type = self.hyper_parameters['optimizer_type']

        """ Network Architecture - 7 layers"""
        self.graph_model = SequentialGraphMol(n_features, batch_size, self.max_atom)
        self.graph_model.add(SGC_LL(n_filters, n_features, batch_size, K=K, activation='relu'))
        self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(SGC_LL(n_filters, n_filters, batch_size, K=K, activation='relu'))
        self.graph_model.add(GraphPoolMol(batch_size))
        self.graph_model.add(DenseMol(final_feature_n, n_filters, activation='relu'))
        self.graph_model.add(GraphGatherMol(batch_size, activation="tanh"))

        """ Classifier """
        self.classifier = MultitaskGraphClassifier(
            self.graph_model,
            len(self.tasks),
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            beta1=beta1,
            beta2=beta2,
            n_feature=final_feature_n
        )

    def train(self):
        assert self.graph_model is not None
        assert self.classifier is not None   # test if network is built

        n_epoch = self.hyper_parameters['n_epoch']  # number of epoch for training

        losses_curve, score_curve = self.classifier.fit(
                                                    self.data['train'],
                                                    self.data['validation'],
                                                    nb_epoch=n_epoch,
                                                    metric=self.metrics,
                                                    transformers=self.transformers)
        self.outputs['losses'] = losses_curve
        self.outputs['score_curve'] = score_curve

    def final_evaluate(self):
        # run and record the classifier's evaluation
        valid_scores, test_scores = dict(), dict()
        valid_scores[self.model_name] = self.classifier.evaluate(
                                                        self.data['validation'],
                                                        self.metrics,
                                                        self.transformers)

        test_scores[self.model_name] = self.classifier.evaluate(
                                                        self.data['testing'],
                                                        self.metrics,
                                                        self.transformers)

        self.outputs['final_score_validation'] = valid_scores
        self.outputs['final_score_testing'] = test_scores

    def get_output(self):
        return self.outputs
