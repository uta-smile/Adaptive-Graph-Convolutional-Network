"""
Implements a multitask classifier.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt

from AGCN.utils.save import log
from AGCN.models.tf_modules.basic_model import Model
from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.tf_modules.graph_topology import merge_dicts
from AGCN.models.tf_modules.evaluation import Evaluator


def get_loss_fn(final_loss):
    # Obtain appropriate loss function
    if final_loss == 'L2':

        def loss_fn(x, t):
            diff = tf.sub(x, t)
            return tf.reduce_sum(tf.square(diff), 0)
    elif final_loss == 'weighted_L2':

        def loss_fn(x, t, w):
            diff = tf.sub(x, t)
            weighted_diff = tf.mul(diff, w)
            return tf.reduce_sum(tf.square(weighted_diff), 0)
    elif final_loss == 'L1':

        def loss_fn(x, t):
            diff = tf.sub(x, t)
            return tf.reduce_sum(tf.abs(diff), 0)
    elif final_loss == 'cross_entropy':

        def loss_fn(x, t, w):
            costs = tf.nn.sigmoid_cross_entropy_with_logits(x, t)
            weighted_costs = tf.mul(costs, w)
            return tf.reduce_sum(weighted_costs)
    elif final_loss == 'softmax_cross_entropy':

        def loss_fn(x, t, w):
            costs = tf.contrib.losses.softmax_cross_entropy(x, t, w)
            return tf.reduce_sum(costs)
    elif final_loss == 'hinge':

        def loss_fn(x, t, w):
            t = tf.mul(2.0, t) - 1
            costs = tf.maximum(0.0, 1.0 - tf.mul(t, x))
            weighted_costs = tf.mul(costs, w)
            return tf.reduce_sum(weighted_costs)
    return loss_fn


class MultitaskGraphClassifier(Model):

    def __init__(self,
                 model,
                 n_tasks,
                 logdir=None,
                 batch_size=50,
                 n_classes=2,
                 final_loss='cross_entropy',
                 learning_rate=.001,
                 optimizer_type="adam",
                 learning_rate_decay_time=50,
                 beta1=.9,
                 beta2=.999,
                 pad_batches=True,
                 verbose=True,
                 n_feature=128,
                 *args,
                 **kwargs):
        super(MultitaskGraphClassifier, self).__init__(*args, **kwargs)
        self.verbose = verbose
        self.n_tasks = n_tasks
        self.final_loss = final_loss
        self.model = model
        self.n_classes = n_classes

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.model.graph, config=config)

        if logdir is not None:
            if not os.path.exists(logdir):
                os.makedirs(logdir)
        else:
            logdir = tempfile.mkdtemp()
        self.logdir = logdir

        with self.model.graph.as_default():
            # Extract model info
            self.batch_size = batch_size
            self.pad_batches = pad_batches
            self.n_feature = n_feature
            # Get graph topology for x
            self.graph_topology = self.model.get_graph_topology()
            ############################################################# DEBUG
            # self.feat_dim = self.model.get_num_output_features()
            # self.feat_dim = n_feat
            ############################################################# DEBUG

            # Raw logit outputs
            self.logits = self.build()
            self.loss_op = self.add_training_loss(self.final_loss, self.logits)     # fit
            self.outputs = self.add_softmax(self.logits)    # predict

            assert type(self.model).__name__ in ['SequentialGraphMol',
                                                 'ResidualGraphMol',
                                                 'ResidualGraphMolResLap',
                                                 'DenseConnectedGraph',
                                                 'DenseConnectedGraphResLap',
                                                 ]
            self.res_L_op = self.model.get_resL_set()
            self.res_W_op = self.model.get_resW_set()
            if type(self.model).__name__ in ['ResidualGraphMolResLap',
                                             'DenseConnectedGraphResLap',
                                             ]:
                self.L_op = self.model.get_laplacian()

            self.learning_rate = learning_rate
            self.T = learning_rate_decay_time
            self.optimizer_type = optimizer_type

            self.optimizer_beta1 = beta1
            self.optimizer_beta2 = beta2

            # Set epsilon
            self.epsilon = 1e-7
            # self.add_optimizer()
            self.global_step = tf.Variable(0, trainable=False)
            # training parameter
            # Initialize
            # self.init_fn = tf.global_variables_initializer()
            # self.sess.run(self.init_fn)

            self._save_path = os.path.join(logdir, 'model.ckpt')

    def build(self):
        # Create target inputs
        self.label_placeholder = tf.placeholder(
            dtype='bool', shape=(None, self.n_tasks), name="label_placeholder")

        # this weight is to mask those unlabeled data and balanced the loss caused by data imbalance
        self.weight_placeholder = tf.placeholder(
            dtype='float32', shape=(None, self.n_tasks), name="weight_placholder")

        feat = self.model.return_outputs()
        output = model_ops.multitask_logits(feat, self.n_tasks, n_feature=self.n_feature)
        return output

    def add_optimizer(self):
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate,
                beta1=self.optimizer_beta1,
                beta2=self.optimizer_beta2,
                epsilon=self.epsilon)
        else:
            raise ValueError("Optimizer type not recognized.")

        # Get train function
        self.train_op = self.optimizer.minimize(self.loss_op)

    def construct_feed_dict(self, X_b, y_b=None, w_b=None):
        """Get initial information about task normalization"""

        n_samples = len(X_b)
        if y_b is None:
            y_b = np.zeros((n_samples, self.n_tasks), dtype=np.bool)
        if w_b is None:
            w_b = np.zeros((n_samples, self.n_tasks), dtype=np.float32)
        targets_dict = {self.label_placeholder: y_b, self.weight_placeholder: w_b}

        # Get graph information
        features_dict = self.graph_topology.batch_to_feed_dict(X_b)

        feed_dict = merge_dicts([targets_dict, features_dict])
        return feed_dict

    def add_training_loss(self, final_loss, logits):
        """Computes loss using logits."""
        loss_fn = get_loss_fn(final_loss)  # Get loss function
        task_losses = []
        # label_placeholder of shape (batch_size, n_tasks). Split into n_tasks
        # tensors of shape (batch_size,)
        task_labels = tf.split(1, self.n_tasks, self.label_placeholder)
        task_weights = tf.split(1, self.n_tasks, self.weight_placeholder)
        for task in range(self.n_tasks):
            task_label_vector = task_labels[task]
            task_weight_vector = task_weights[task]
            # Convert the labels into one-hot vector encodings.
            one_hot_labels = tf.to_float(
                tf.one_hot(tf.to_int32(tf.squeeze(task_label_vector)), 2))  # 0,1 to one_hot
            # Since we use tf.nn.softmax_cross_entropy_with_logits note that we pass in
            # un-softmaxed logits rather than softmax outputs.
            task_loss = loss_fn(logits[task], one_hot_labels, task_weight_vector)
            task_losses.append(task_loss)
        # It's ok to divide by just the batch_size rather than the number of nonzero
        # examples (effect averages out)
        total_loss = tf.add_n(task_losses)
        total_loss = tf.div(total_loss, self.batch_size)
        return total_loss

    def add_softmax(self, outputs):
        """Replace logits with softmax outputs."""
        softmax = []
        with tf.name_scope('inference'):
            for i, logits in enumerate(outputs):
                softmax.append(tf.nn.softmax(logits, name='softmax_%d' % i))
        return softmax

    def fit(self,
            train_data,
            val_data,
            nb_epoch,
            metrics=None,
            transformers=None,
            max_checkpoints_to_keep=5,
            log_every_N_batches=50):
        # Perform the optimization
        log("Training for %d epochs" % nb_epoch, self.verbose)

        with self.model.graph.as_default():

            # setup the exponential decayed learning rate
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       self.global_step, self.T, 0.7, staircase=True)

            # construct train_op with decayed rate
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_op, global_step=self.global_step)
            # initialize the graph
            self.init_fn = tf.global_variables_initializer()
            self.sess.run(self.init_fn)

            loss_curve = []
            scores_curves = {metric.name: [] for metric in metrics}  # multiple metrics, task-averaged scores

            with self.sess.as_default():
                for epoch in range(nb_epoch):
                    log("Starting epoch %d" % epoch, self.verbose)

                    for batch_num, (X_b, y_b, w_b, ids_b) in enumerate(
                            train_data.iterbatches(self.batch_size, pad_batches=self.pad_batches)):

                        # print(batch_num)
                        """ these network models contains SGC_LL layer,
                        which also return updated residual Laplacian"""
                        if type(self.model).__name__ in ['ResidualGraphMolResLap',
                                                         'DenseConnectedGraphResLap',
                                                         ]:
                            _, loss_val, res_L, res_W, L = self.sess.run(
                                [self.train_op, self.loss_op, self.res_L_op, self.res_W_op, self.L_op],
                                feed_dict=self.construct_feed_dict(X_b, y_b=y_b, w_b=w_b))
                        else:
                            _, loss_val, res_L, res_W = self.sess.run(
                                [self.train_op, self.loss_op, self.res_L_op, self.res_W_op],
                                feed_dict=self.construct_feed_dict(X_b, y_b=y_b, w_b=w_b))

                        if batch_num % 10 == 0:
                            print('loss = ' + str(loss_val))
                            loss_curve.append(loss_val)
                        # if epoch == 10:
                            # self.watch_batch(X_b, ids_b, res_L, L)

                    if epoch % 5 == 0:
                        scores = self.evaluate(val_data, metrics, transformers)
                        for metric in metrics:
                            scores_curves[metric.name].append(scores[metric.name])

        return loss_curve, scores_curves

    def watch_batch(self, X_b, ids_b, L_update):
        # visualize the intrinsic Laplacian and residual Laplacian updated during training
        """watch on specific group of samples, each epoch display once"""
        # test if length of L_update == batch size * SGC layer number
        assert len(L_update) % self.batch_size == 0
        conv_layer_n = int(len(L_update)/self.batch_size)
        all_L = np.asarray(L_update).reshape((conv_layer_n, self.batch_size)).T
        shape = all_L.size
        for graph in X_b:
            X = graph.node_features

        print(shape)

    def save(self):
        """
        No-op since this model doesn't currently support saving...
        """
        raise NotImplementedError

    def predict(self, dataset, transformers=[], **kwargs):
        """Wraps predict to set batch_size/padding."""
        return super(MultitaskGraphClassifier, self).predict(
            dataset, transformers, batch_size=self.batch_size)

    def predict_proba(self, dataset, transformers=[], n_classes=2, **kwargs):
        """Wraps predict_proba to set batch_size/padding."""
        return super(MultitaskGraphClassifier, self).predict_proba(
            dataset, transformers, n_classes=n_classes, batch_size=self.batch_size)

    def predict_on_batch(self, X):
        """Return model output for the provided input.
        """
        if self.pad_batches:
            # make sure X size == batch size
            X = self.pad_graphs(self.batch_size, X)

        # run eval data through the model
        with self.sess.as_default():
            feed_dict = self.construct_feed_dict(X)
            # Shape (n_samples, n_tasks)
            batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

        outputs = np.zeros((self.batch_size, self.n_tasks))
        for task, output in enumerate(batch_outputs):
            outputs[:, task] = np.argmax(output, axis=1)
        return outputs

    def predict_proba_on_batch(self, X, n_classes=2):
        """Returns class probabilities on batch"""
        # run eval data through the model
        if self.pad_batches:
            X = self.pad_graphs(self.batch_size, X)

        with self.sess.as_default():
            feed_dict = self.construct_feed_dict(X)
            batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

        n_samples = len(X)
        outputs = np.zeros((n_samples, self.n_tasks, n_classes))
        for task, output in enumerate(batch_outputs):
            outputs[:, task, :] = output
        return outputs

    def get_num_tasks(self):
        """Needed to use Model.predict() from superclass."""
        return self.n_tasks

    @staticmethod
    def pad_graphs(batch_size, X_b):
        """Pads a batch of features to have precisely batch_size elements.

        Version of pad_batch for use at prediction time.
        """
        num_samples = len(X_b)
        if num_samples == batch_size:
            return X_b
        else:
            # By invariant of when this is called, can assume num_samples > 0
            # and num_samples < batch_size
            if len(X_b.shape) > 1:
                feature_shape = X_b.shape[1:]
                X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
            else:
                X_out = np.zeros((batch_size,), dtype=X_b.dtype)

            # Fill in batch arrays
            start = 0
            while start < batch_size:
                num_left = batch_size - start
                if num_left < num_samples:
                    increment = num_left
                else:
                    increment = num_samples
                X_out[start:start + increment] = X_b[:increment]
                start += increment
            return X_out

    def evaluate(self, dataset, metrics, transformers=[], per_task_metrics=False):
        """
        Evaluates the performance of this model on specified dataset.

        Parameters
        ----------
        dataset: dc.data.Dataset
          Dataset object.
        metrics: deepchem.metrics.Metric
          Evaluation metric
        transformers: list
          List of deepchem.transformers.Transformer
        per_task_metrics: bool
          If True, return per-task scores.

        Returns
        -------
        dict
          Maps tasks to scores under metric.
        """
        if not isinstance(metrics, list):
            metrics = [metrics]

        evaluator = Evaluator(self, dataset, transformers)

        if not per_task_metrics:
            # only task-averaged scores, dict format
            scores = evaluator.compute_model_performance(metrics)
            return scores
        else:
            scores, per_task_scores = evaluator.compute_model_performance(
                metrics, per_task_metrics=per_task_metrics)
            return scores, per_task_scores
