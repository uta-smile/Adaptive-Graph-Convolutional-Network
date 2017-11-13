"""
Implements a part segmentation module for graph AGCN.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tempfile
import matplotlib.pyplot as plt
import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys

from AGCN.utils.save import log
from AGCN.models.tf_modules.basic_model import Model
from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.tf_modules.graph_topology import merge_dicts
from AGCN.models.tf_modules.evaluation import Evaluator
from AGCN.models.tf_modules.multitask_classifier import get_loss_fn


LEARNING_RATE_CLIP = 1e-5


class PartSegmentation(object):
    """
    Class of a part segmentation network
    """

    def __init__(self,
                 model,
                 logdir=None,
                 batch_size=50,
                 n_classes=2,
                 num_point=1024,
                 learning_rate=.001,
                 optimizer_type="adam",
                 decay_step=2e4,
                 decay_rate=0.7,
                 beta1=.9,
                 beta2=.999,
                 pad_batches=True,
                 verbose=True,
                 n_feature=128):

        self.model = model
        with self.model.graph.as_default():

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(graph=self.model.graph, config=config)

            if logdir is not None:
                if not os.path.exists(logdir):
                    os.makedirs(logdir)
            else:
                logdir = tempfile.mkdtemp()
            self.logdir = logdir

            self.batch_size = batch_size
            self.pad_batches = pad_batches
            self.n_feature = n_feature
            self.n_tasks = 1  # single task
            self.n_classes = n_classes
            self.num_point = num_point
            self.verbose = verbose
            # Get graph topology for x

            self.graph_topology = self.model.get_graph_topology()

            self.decay_step = decay_step   # Decay Step
            self.decay_rate = decay_rate

            self.optimizer_type = optimizer_type
            self.optimizer_beta1 = beta1
            self.optimizer_beta2 = beta2
            self.epsilon = 1e-7

            self.global_step = tf.Variable(0, trainable=False)

            self._save_path = os.path.join(logdir, 'model.ckpt')
            self.saver = tf.train.Saver()

            self.decay_learning_rate = tf.train.exponential_decay(
                learning_rate,  # base learning rate
                self.global_step * self.batch_size,  # global_var indicating the number of steps
                self.decay_step,  # step size
                self.decay_rate,  # decay rate
                staircase=True  # Stair-case or continuous decreasing
            )
            self.learning_rate = tf.maximum(self.decay_learning_rate, LEARNING_RATE_CLIP)
            self.lr_op = tf.summary.scalar('learning_rate', self.learning_rate)

            # accept the raw input
            self._placeholder_inputs()
            # Raw model outputs tensors from the network model
            self._build()
            # training loss ops
            self._add_training_loss()  # for fitting

            # inference ops
            self.label_pred, self.seg_pred = self._add_inference()  # inference op

            self._train_observer()
            self._add_optimizer()

            # initialize the graph
            self.init_fn = tf.global_variables_initializer()
            self.sess.run(self.init_fn)

            # tf writer
            self.train_writer = tf.summary.FileWriter(logdir + '/train', self.sess.graph)
            self.test_writer = tf.summary.FileWriter(logdir + '/test')

    def _placeholder_inputs(self):
        self.pointclouds_ph = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_point, 3))
        self.input_label_ph = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_classes))
        self.labels_ph = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.seg_ph = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_point))

    def _build(self):
        self.label_pred_feat = self.model.classification_outputs()
        self.seg_pred_feat = self.model.segmentation_outputs()

    def _add_training_loss(self):
        self.per_instance_label_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.label_pred_feat,
            labels=self.labels_ph)

        self.label_loss_op = tf.reduce_mean(self.per_instance_label_loss_op)  # sample_average loss on label prediction

        self.per_instance_seg_loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.seg_pred_feat, labels=self.seg_ph),
            axis=1,
        )
        self.seg_loss_op = tf.reduce_mean(self.per_instance_seg_loss_op)

        self.total_loss_op = self.seg_loss_op + self.label_loss_op

    def _add_inference(self):
        per_instance_seg_pred_res = tf.argmax(self.seg_pred_feat, 2)
        per_instance_label_pred_res = tf.argmax(self.label_pred_feat, 1)

        return per_instance_label_pred_res, per_instance_seg_pred_res

    def _train_observer(self):
        """
        training, evaluation result, Scalar tensor
        """

        self.total_training_loss_ph = tf.placeholder(tf.float32, shape=())
        self.total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

        self.label_training_loss_ph = tf.placeholder(tf.float32, shape=())
        self.label_testing_loss_ph = tf.placeholder(tf.float32, shape=())

        self.seg_training_loss_ph = tf.placeholder(tf.float32, shape=())
        self.seg_testing_loss_ph = tf.placeholder(tf.float32, shape=())

        self.label_training_acc_ph = tf.placeholder(tf.float32, shape=())
        self.label_testing_acc_ph = tf.placeholder(tf.float32, shape=())
        self.label_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

        self.seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
        self.seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
        self.seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

        self.total_train_loss_sum_op = tf.summary.scalar('total_training_loss', self.total_training_loss_ph)
        self.total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', self.total_testing_loss_ph)

        self.label_train_loss_sum_op = tf.summary.scalar('label_training_loss', self.label_training_loss_ph)
        self.label_test_loss_sum_op = tf.summary.scalar('label_testing_loss', self.label_testing_loss_ph)

        self.seg_train_loss_sum_op = tf.summary.scalar('seg_training_loss', self.seg_training_loss_ph)
        self.seg_test_loss_sum_op = tf.summary.scalar('seg_testing_loss', self.seg_testing_loss_ph)

        self.label_train_acc_sum_op = tf.summary.scalar('label_training_acc', self.label_training_acc_ph)
        self.label_test_acc_sum_op = tf.summary.scalar('label_testing_acc', self.label_testing_acc_ph)
        self.label_test_acc_avg_cat_op = tf.summary.scalar('label_testing_acc_avg_cat',
                                                           self.label_testing_acc_avg_cat_ph)

        self.seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', self.seg_training_acc_ph)
        self.seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', self.seg_testing_acc_ph)
        self.seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', self.seg_testing_acc_avg_cat_ph)

    def _add_optimizer(self):
        train_variables = tf.trainable_variables()
        trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = trainer.minimize(
            self.total_loss_op,
            var_list=train_variables,
            global_step=self.global_step)

    def print_log(self, flog, data):
        print(data)
        flog.write(data + '\n')

    def construct_feed_dict(self, X_b, y_b, seg_truth_b, one_hot_y_b):
        """Get initial information about task normalization"""
        assert len(X_b) == self.batch_size

        targets_dict = {
            self.input_label_ph: one_hot_y_b,
            self.labels_ph: y_b,
            self.seg_ph: seg_truth_b,
        }

        # Get graph information
        features_dict = self.graph_topology.batch_to_feed_dict(X_b)

        feed_dict = merge_dicts([targets_dict, features_dict])
        return feed_dict

    def fit(self,
            train_data,
            test_data,
            nb_epoch,
            ):
        """
        Train the segmentation network based on AGCN family of models
        :param train_data:
        :param test_data:
        :param nb_epoch:
        :return:
        """
        with self.model.graph.as_default():
            with self.sess.as_default():
                flog = open(os.path.join(self.logdir, 'log.txt'), 'w')
                num_data = train_data['X'].shape[0]

                """load data"""
                print ("Loading training data....")
                X = train_data['X']
                y = train_data['y']
                seg_mask = train_data['seg_mask']
                one_hot_y = train_data['one_hot_y']

                for epoch_num in range(nb_epoch):
                    print ("Training at Epoch %d" % epoch_num)

                    total_loss = 0.0
                    total_label_loss = 0.0
                    total_seg_loss = 0.0
                    total_label_acc = 0.0
                    total_seg_acc = 0.0

                    total_num_batch = int(num_data // self.batch_size)
                    for batch_num in range(total_num_batch):
                        begidx = batch_num * self.batch_size
                        endidx = (batch_num + 1) * self.batch_size
                        X_b = X[begidx: endidx]
                        y_b = y[begidx: endidx]
                        seg_mask_b = seg_mask[begidx: endidx, :]
                        one_hot_y_b = one_hot_y[begidx: endidx, :]

                        _, loss_val, label_loss_val, seg_loss_val, per_instance_label_loss_val, \
                            per_instance_seg_loss_val, label_pred_val, seg_pred_val, pred_label_res, pred_seg_res \
                            = self.sess.run(
                                            [self.train_op,
                                             self.total_loss_op,
                                             self.label_loss_op,
                                             self.seg_loss_op,
                                             self.per_instance_label_loss_op,
                                             self.per_instance_seg_loss_op,
                                             self.label_pred_feat,
                                             self.seg_pred_feat,
                                             self.label_pred,
                                             self.seg_pred],
                                            feed_dict=self.construct_feed_dict(X_b, y_b, seg_mask_b, one_hot_y_b))

                        """segmentation metrics"""
                        per_instance_part_acc = np.mean(np.float32(pred_seg_res == seg_mask_b), axis=1)
                        average_part_acc = np.mean(per_instance_part_acc)   # average over batch

                        " accumulated loss from all batches"
                        total_loss += loss_val
                        total_label_loss += label_loss_val
                        total_seg_loss += seg_loss_val

                        total_label_acc += np.mean(np.float32(pred_label_res == y_b))
                        total_seg_acc += average_part_acc

                    """each epoch, get batch-average metrics Loss + Score """
                    total_loss = total_loss * 1.0 / batch_num
                    total_label_loss = total_label_loss * 1.0 / batch_num
                    total_seg_loss = total_seg_loss * 1.0 / batch_num
                    total_label_acc = total_label_acc * 1.0 / batch_num
                    total_seg_acc = total_seg_acc * 1.0 / batch_num

                    lr, train_loss_sum, train_label_acc_sum, \
                        train_label_loss_sum, train_seg_loss_sum, train_seg_acc_sum = self.sess.run(
                            [self.lr_op,
                             self.total_train_loss_sum_op,
                             self.label_train_acc_sum_op,
                             self.label_train_loss_sum_op,
                             self.seg_train_loss_sum_op,
                             self.seg_train_acc_sum_op],
                            feed_dict={self.total_training_loss_ph: total_loss,
                                       self.label_training_loss_ph: total_label_loss,
                                       self.seg_training_loss_ph: total_seg_loss,
                                       self.label_training_acc_ph: total_label_acc,
                                       self.seg_training_acc_ph: total_seg_acc}
                        )

                    "use tensor board writer"
                    self.train_writer.add_summary(lr, epoch_num)
                    self.train_writer.add_summary(train_loss_sum, epoch_num)
                    self.train_writer.add_summary(train_label_acc_sum, epoch_num)
                    self.train_writer.add_summary(train_label_loss_sum, epoch_num)
                    self.train_writer.add_summary(train_seg_loss_sum, epoch_num)
                    self.train_writer.add_summary(train_seg_acc_sum, epoch_num)

                    "write result to log"
                    self.print_log(flog, '\tTraining Total Mean_loss: %f' % total_loss)
                    self.print_log(flog, '\t\tTraining Label Mean_loss: %f' % total_label_loss)
                    self.print_log(flog, '\t\tTraining Label Accuracy: %f' % total_label_acc)
                    self.print_log(flog, '\t\tTraining Seg Mean_loss: %f' % total_seg_loss)
                    self.print_log(flog, '\t\tTraining Seg Accuracy: %f' % total_seg_acc)


