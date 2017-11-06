from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import csv
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime

from AGCN.models.tf_modules.tf_graphs import SequentialGraphMol
from AGCN.models.layers import DenseMol, SGC_LL, GraphGatherMol, GraphPoolMol, BatchNormalization
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

    def train(self):
        assert self.graph_model is not None
        assert self.classifier is not None   # test if network is built
        print("Start Training ...... \n\n")
        n_epoch = self.hyper_parameters['n_epoch']  # number of epoch for training

        losses_curve, score_curve = self.classifier.fit(
                                                    self.data['train'],
                                                    self.data['validation'],
                                                    nb_epoch=n_epoch,
                                                    metrics=self.metrics,
                                                    transformers=self.transformers)
        self.outputs['losses'] = losses_curve
        self.outputs['score_curve'] = score_curve
        self.save_training_data()

    def save_training_data(self):
        assert os.path.exists(self.save_dir)
        assert len(self.outputs['losses']) > 0

        # save loss and scores curves into csv file in local disk
        with open(os.path.join(self.save_dir, self.saved_csv_name), 'a') as f:
            writer = csv.writer(f)
            # write time
            time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            output_line = [time_now]
            writer.writerow(output_line)

            # record some hyper-parameters for tuning
            writer.writerow([self.hyper_parameters['learning_rate']] + self.hyper_parameters['l_n_filters'])

            # record training curves
            output_line = [self.model_name, self.data_name, 'training', 'epoch_n:', self.hyper_parameters['n_epoch']]
            writer.writerow(output_line)
            output_line = ['loss_curve'] + self.outputs['losses']
            writer.writerow(output_line)

            for metric in self.metrics:
                output_line = [metric.name] + self.outputs['score_curve'][metric.name]
                writer.writerow(output_line)
        print('Training data/curves saved!')

    def final_evaluate(self):
        # run and record the classifier's evaluation
        valid_scores = self.classifier.evaluate(
                                                self.data['validation'],
                                                self.metrics,
                                                self.transformers)

        test_scores = self.classifier.evaluate(
                                                self.data['testing'],
                                                self.metrics,
                                                self.transformers)

        self.outputs['final_score_validation'] = valid_scores
        self.outputs['final_score_testing'] = test_scores

        self.save_evaluation_testing_data()

    def save_evaluation_testing_data(self):
        assert os.path.exists(self.save_dir)
        assert len(self.outputs['final_score_validation']) > 0

        writer = csv.writer(open(os.path.join(self.save_dir, self.saved_csv_name), 'a'))

        output_line = [self.model_name, self.data_name, 'final evaluation/testing scores']
        writer.writerow(output_line)

        for metric in self.metrics:
            output_line = ['on evaluation:', metric.name, self.outputs['final_score_validation'][metric.name]]
            writer.writerow(output_line)
            output_line = ['on testing:', metric.name, self.outputs['final_score_testing'][metric.name]]
            writer.writerow(output_line)
        print('Evaluation, testing result saved!')

    def get_outputs(self):
        return self.outputs

    def print_save_dir(self):
        print('result scores saved at:\n', self.save_dir)

    def return_save_dir(self):
        return self.save_dir

    def plot_loss_curve(self):
        # plot loss curve

        # check if curves are saved
        assert os.path.exists(os.path.join(self.save_dir, self.saved_csv_name))

        try:
            reader = csv.reader(open(os.path.join(self.save_dir, self.saved_csv_name)), delimiter=str(','))
            for id, row in enumerate(reader):
                if id == 0 and row[2] == 'training':
                    print('model: {}'.format(row[0]))
                    print('data: {}'.format(row[1]))
                    label_curve = str(row[0] + row[1])

                if 'loss_curve' in row and len(row) > 1:
                    assert label_curve is not None
                    print('plotting....')
                    plt.figure(0)
                    plt.plot(range(len(row[1:])), map(float, row[1:]), marker='o', color='c', label=label_curve,
                             linewidth=3.0)
                    plt.show()
                    if self.save_fig:
                        plt.savefig(os.path.join(self.save_dir,
                                                 self.model_name + '_' + self.data_name + '_loss.png')
                                    )
                        print('Figure saved at:', self.save_dir)
                    return
            else:
                print('No Loss curve data saved! Check file again:', self.saved_csv_name)

        except ValueError:
            print('Failed to open figure from : ', self.saved_csv_name)

    def plot_score_curves(self, metric=None):
        # check if curves are saved
        assert os.path.exists(os.path.join(self.save_dir, self.saved_csv_name))

        if metric is not None and metric not in self.metric_names:
            raise ValueError('input metric is not calculated! ')

        if metric:
            # plot curve of specific metric
            reader = csv.reader(open(os.path.join(self.save_dir, self.saved_csv_name)), delimiter=str(','))
            for id, row in enumerate(reader):

                if metric in row and len(row) > 1:
                    print('plotting {}....'.format(metric))
                    plt.figure(0)
                    plt.plot(range(len(row[1:])), map(float, row[1:]), marker='o', color='c',
                             linewidth=3.0)
                    plt.show()
                    if self.save_fig:
                        plt.savefig(os.path.join(self.save_dir,
                                                 self.model_name + '_' + self.data_name + '_' + str(metric) + '.png'))
                        print('Figure saved at:', self.save_dir)
                    return
            else:
                print('No {} curve data saved! Check file again.'.format(metric))
        else:
            # plot all metric scores
            reader = csv.reader(open(os.path.join(self.save_dir, self.saved_csv_name)), delimiter=',')
            fig_id = 0
            for id, row in enumerate(reader):
                if row[0] in self.metric_names and len(row) > 1:
                    print('plotting {}....'.format(str(row[0])))
                    plt.figure(fig_id)
                    fig_id += 1
                    plt.plot(range(len(row[1:])), map(float, row[1:]), marker='o', color='c',
                             linewidth=3.0)
                    plt.show()
                    if self.save_fig:
                        plt.savefig(os.path.join(self.save_dir,
                                                 self.model_name + '_' + self.data_name + '_' + str(row[0]) + '.png'))
                        print('Figure saved at:', self.save_dir)
            return


class LongAGCN(SimpleAGCN):
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
