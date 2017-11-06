from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tensorflow as tf
import tempfile

from AGCN.utils.save import log
from AGCN.models.tf_modules.basic_model import Model
from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.tf_modules.multitask_classifier import MultitaskGraphClassifier
from AGCN.models.tf_modules.graph_topology import merge_dicts
from AGCN.models.tf_modules.multitask_classifier import get_loss_fn
from AGCN.models.tf_modules.evaluation import Evaluator

import pylab
# import matplotlib.pyplot as plt


class MultitaskGraphRegressor(Model):
    def __init__(self,
                 model,
                 n_tasks,
                 logdir=None,
                 batch_size=50,
                 final_loss='weighted_L2',
                 learning_rate=.001,
                 optimizer_type="adam",
                 learning_rate_decay_time=50,
                 beta1=.9,
                 beta2=.999,
                 pad_batches=True,
                 verbose=True,
                 n_feature=128):

        self.verbose = verbose
        self.n_tasks = n_tasks
        self.final_loss = final_loss
        self.model = model

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
            # self.feat_dim = n_feat

            # Building outputs
            self.outputs = self.build()  # no logistic but a fully connected layer to generate prediction for each task
            self.loss_op = self.add_training_loss(self.final_loss, self.outputs)

            assert type(self.model).__name__ in ['SequentialGraphMol',
                                                 'ResidualGraphMol',
                                                 'ResidualGraphMolResLap',
                                                 'DenseConnectedGraph',
                                                 'DenseConnectedGraphResLap',
                                                 ]
            self.res_L_op = self.model.get_resL_set()
            self.res_W_op = self.model.get_resW_set()
            if type(self.model).__name__ in ['ResidualGraphMolResLap', 'DenseConnectedGraphResLap']:
                self.L_op = self.model.get_laplacian()

            self.learning_rate = learning_rate
            self.T = learning_rate_decay_time
            self.optimizer_type = optimizer_type

            self.optimizer_beta1 = beta1
            self.optimizer_beta2 = beta2

            # Set epsilon
            self.epsilon = 1e-7
            # self.add_optimizer()

            # Initialize
            # self.init_fn = tf.initialize_all_variables()
            # self.sess.run(self.init_fn)
            self.global_step = tf.Variable(0, trainable=False)
            # Path to save checkpoint files, which matches the
            # replicated supervisor's default path.
            self._save_path = os.path.join(logdir, 'model.ckpt')

    def build(self):
        # Create target inputs
        self.label_placeholder = tf.placeholder(
            dtype='float32', shape=(None, self.n_tasks), name="label_placeholder")
        self.weight_placeholder = tf.placeholder(
            dtype='float32', shape=(None, self.n_tasks), name="weight_placholder")

        feat = self.model.return_outputs()
        # feat_size = feat.get_shape()[-1].value
        feat_size = self.n_feature
        outputs = []
        for task in range(self.n_tasks):
            outputs.append(
                tf.squeeze(
                    model_ops.fully_connected_layer(
                        tensor=feat,
                        size=1,
                        weight_init=tf.truncated_normal(
                            shape=[feat_size, 1], stddev=0.01),
                        bias_init=tf.constant(value=0., shape=[1]))))
        return outputs

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

    def add_training_loss(self, final_loss, outputs):
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
            task_loss = loss_fn(outputs[task],
                                tf.squeeze(task_label_vector),
                                tf.squeeze(task_weight_vector))
            task_losses.append(task_loss)
        # It's ok to divide by just the batch_size rather than the number of nonzero
        # examples (effect averages out)
        total_loss = tf.add_n(task_losses)
        total_loss = tf.div(total_loss, self.batch_size)
        return total_loss

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
                        if type(self.model).__name__ in ['ResidualGraphMolResLap', 'DenseConnectedGraphResLap']:
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

    def save(self):
        """
        No-op since this model doesn't currently support saving...
        """
        pass

    def print_lap_smile(self, smile, ids_s, L, epoch):
        save_dir = os.path.join(os.environ['HOME'], 'experiment_result/delayney_reg_varying_laplacian')
        for i, s in enumerate(ids_s.tolist()):
            if s in smile:
                fig = pylab.figure()
                L_layers = [L[i], L[i+int(self.batch_size)]]
                L_layers = [L[i]]
                # plt.imshow(L_layers[0], cmap='hot', interpolation='nearest')
                # plt.savefig(os.path.join(save_dir, 'w_conv1_ep_' + str(epoch) + '.png'))
                pylab.imshow(L_layers[0], cmap='hot', interpolation='nearest')

                # pylab.colorbar()
                fig.savefig(os.path.join(save_dir, 'ML_orig_ep_l' + str(epoch) + '.png'), pad_inches=0, bbox_inches='tight')

    def predict(self, dataset, transformers=[], **kwargs):
        """Wraps predict to set batch_size/padding."""
        return super(MultitaskGraphRegressor, self).predict(
            dataset, transformers, batch_size=self.batch_size)

    def predict_on_batch(self, X):
        """Return model output for the provided input.
        """
        if self.pad_batches:
            X = MultitaskGraphClassifier.pad_graphs(self.batch_size, X)

        # run eval data through the model
        with self.sess.as_default():
            feed_dict = self.construct_feed_dict(X)
            # Shape (n_samples, n_tasks)
            batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

        outputs = np.zeros((self.batch_size, self.n_tasks))
        for task, output in enumerate(batch_outputs):
            outputs[:, task] = output
        return outputs

    def get_num_tasks(self):
        """Needed to use Model.predict() from superclass."""
        return self.n_tasks

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
