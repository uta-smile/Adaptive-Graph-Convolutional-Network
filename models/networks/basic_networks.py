from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import os
from AGCN.models.tf_modules import MultitaskGraphClassifier, MultitaskGraphRegressor


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
                 save_fig=True,
                 ):

        self.data = {'train': train_data, 'validation': valid_data, 'testing': test_data}

        assert hyper_parameters is not None
        self.hyper_parameters = hyper_parameters

        self.max_atom = max_atom
        self.tasks = tasks
        self.metrics = metrics
        self.metric_names = [m.name for m in self.metrics]

        self.transformers = transformers
        self.seed = self.hyper_parameters['seed']

        self.graph_model = None     # network architecture
        """
         construct network
         """
        self.construct_network()

        self.classifier = None   # network model for training and evaluation
        self.tasks_type = self.hyper_parameters['task_type']     # regression / classification / segmentation

        """ Define Classifier """
        if self.tasks_type == "classification":
            self.classifier = MultitaskGraphClassifier(
                self.graph_model,
                len(self.tasks),
                batch_size=self.hyper_parameters['batch_size'],
                learning_rate=self.hyper_parameters['learning_rate'],
                optimizer_type=self.hyper_parameters['optimizer_type'],
                beta1=self.hyper_parameters['optimizer_beta1'],
                beta2=self.hyper_parameters['optimizer_beta2'],
                n_feature=self.hyper_parameters['final_feature_n']
            )
        elif self.tasks_type == "regression":
            self.classifier = MultitaskGraphRegressor(
                self.graph_model,
                len(self.tasks),
                batch_size=self.hyper_parameters['batch_size'],
                learning_rate=self.hyper_parameters['learning_rate'],
                optimizer_type=self.hyper_parameters['optimizer_type'],
                beta1=self.hyper_parameters['optimizer_beta1'],
                beta2=self.hyper_parameters['optimizer_beta2'],
                n_feature=self.hyper_parameters['final_feature_n']
            )

        self.model_name = self.hyper_parameters['model_name']
        self.data_name = self.hyper_parameters['data_name']
        self.saved_csv_name = self.model_name + '_' + self.data_name + '.csv'

        self.outputs = {}
        # one experiment, one save file
        self.save_dir = self.hyper_parameters['save_dir']
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_fig = save_fig

    def construct_network(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def final_evaluate(self):
        raise NotImplementedError

