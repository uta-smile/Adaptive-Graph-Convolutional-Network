"""
Implements a single task classifier.
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
from AGCN.models.tf_modules.multitask_classifier import get_loss_fn


class SingletaskGraphClassifier(Model):
    def __init__(self,
                 model,
                 logdir=None,
                 batch_size=50,
                 n_classes=2,
                 final_loss='softmax_cross_entropy',
                 learning_rate=.001,
                 optimizer_type="adam",
                 learning_rate_decay_time=2e5,
                 beta1=.9,
                 beta2=.999,
                 pad_batches=True,
                 verbose=True,
                 n_feature=128,
                 *args,
                 **kwargs):
        super(SingletaskGraphClassifier, self).__init__(*args, **kwargs)

        self.n_tasks = 1    # single task
        self.n_classes = n_classes
        self.verbose = verbose
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

            self.batch_size = batch_size
            self.pad_batches = pad_batches
            self.n_feature = n_feature
            # Get graph topology for x
            self.graph_topology = self.model.get_graph_topology()

            # Raw logit outputs from the network model
            self.logits = self._build()
            # training loss
            self.loss_op = self._add_training_loss(self.final_loss, self.logits)  # fit
            # inference score
            self.outputs = self._add_softmax(self.logits)  # predict

            # must be one of the known network model
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
                # those two has one more tensor
                self.L_op = self.model.get_laplacian()

            self.T = learning_rate_decay_time
            self.optimizer_type = optimizer_type

            self.optimizer_beta1 = beta1
            self.optimizer_beta2 = beta2

            self.epsilon = 1e-7
            self.global_step = tf.Variable(0, trainable=False)
            self._save_path = os.path.join(logdir, 'model.ckpt')

            # setup the exponential decayed learning rate
            self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                            self.global_step * batch_size, self.T, 0.7, staircase=True)

            self.learning_rate = tf.maximum(self.learning_rate, 0.00001)
            self.lr_op = tf.summary.scalar('learning_rate', self.learning_rate)
            # construct train_op with decayed rate
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op,
                                                                                global_step=self.global_step)
            # initialize the graph
            self.init_fn = tf.global_variables_initializer()
            self.sess.run(self.init_fn)

            self.train_writer = tf.summary.FileWriter(logdir + '/train', self.sess.graph)
            self.test_writer = tf.summary.FileWriter(logdir + '/test')

    def _build(self):
        # Create target inputs
        self.label_placeholder = tf.placeholder(
            dtype='bool', shape=(self.batch_size, ), name="label_placeholder")

        # this weight is to mask those unlabeled data and reduce the loss caused by data imbalance
        self.weight_placeholder = tf.placeholder(
            dtype='float32', shape=(self.batch_size, ), name="weight_placholder")

        feat = self.model.return_outputs()
        output = model_ops.logits(feat, num_classes=self.n_classes, n_feature=self.n_feature, dropout_prob=0.3)
        return output

    def _add_training_loss(self, final_loss, logits):
        """Computes loss using logits with final_loss function."""

        loss_fn = get_loss_fn(final_loss)  # Get loss function
        # task_losses = []
        # label_placeholder of shape (batch_size, n_tasks). Split into n_tasks
        # tensors of shape (batch_size,)
        labels = self.label_placeholder
        labels = tf.one_hot(tf.to_int32(labels), self.n_classes)
        sample_weights = self.weight_placeholder

        # task_label_vector = task_labels
        # task_weight_vector = task_weights
        # Convert the labels into one-hot vector encodings.
        labels = tf.to_float(tf.reshape(labels, (self.batch_size, self.n_classes)))  # 0,1 to one_hot
        # Since we use tf.nn.labels note that we pass in
        # un-softmaxed logits rather than softmax outputs.
        # preds = tf.argmax(logits, axis=1).astype(tf.float32)
        total_loss = loss_fn(logits, labels, sample_weights)
        # task_losses.append(task_loss)
        # It's ok to divide by just the batch_size rather than the number of nonzero
        # examples (effect averages out)
        # total_loss = tf.add_n(task_losses)
        # total_loss = tf.div(total_loss, self.batch_size)
        return total_loss

    def _add_softmax(self, logits):
        """Replace logits with softmax outputs."""
        with tf.name_scope('inference'):
            softmax = tf.nn.softmax(logits, name='softmax')
        return softmax

    def construct_feed_dict(self, X_b, y_b=None, w_b=None):
        """Get initial information about task normalization"""
        assert len(X_b) == self.batch_size
        n_samples = len(X_b)
        if y_b is None:
            y_b = np.zeros((n_samples, ), dtype=np.bool)
        if w_b is None:
            w_b = np.zeros((n_samples, ), dtype=np.float32)
        targets_dict = {self.label_placeholder: y_b, self.weight_placeholder: w_b}

        # Get graph information
        features_dict = self.graph_topology.batch_to_feed_dict(X_b)

        feed_dict = merge_dicts([targets_dict, features_dict])
        return feed_dict

    def fit(self,
            train_data,
            test_data,
            nb_epoch,
            metrics=None,
            transformers=None,
            max_checkpoints_to_keep=5,
            log_every_N_batches=50):

        assert metrics is not None

        # Perform the optimization
        log("Training for %d epochs" % nb_epoch, self.verbose)
        with self.model.graph.as_default():

            loss_curve = []
            scores_curves = {metric.name: [] for metric in metrics}  # multiple metrics, task-averaged scores

            num_data = train_data['X'].shape[0]

            """load data"""
            print("Loading training data....")
            X = train_data['X']
            y = train_data['y']
            w = train_data['w']
            total_num_batch = int(num_data // self.batch_size)

            with self.sess.as_default():
                for epoch in range(nb_epoch):
                    log("Starting epoch %d" % epoch, self.verbose)

                    for batch_num in range(total_num_batch):
                        # print(batch_num)
                        begidx = batch_num * self.batch_size
                        endidx = (batch_num + 1) * self.batch_size
                        X_b = X[begidx: endidx]
                        y_b = y[begidx: endidx]
                        w_b = w[begidx: endidx]

                        """ these network models contains SGC_LL layer,
                        which also return updated residual Laplacian"""
                        if type(self.model).__name__ in ['ResidualGraphMolResLap',
                                                         'DenseConnectedGraphResLap',
                                                         ]:
                            _, loss_val, res_L, res_W, L, lr = self.sess.run(
                                [self.train_op, self.loss_op, self.res_L_op, self.res_W_op, self.L_op, self.lr_op
                                 ],
                                feed_dict=self.construct_feed_dict(X_b, y_b=y_b, w_b=w_b))
                        else:
                            _, loss_val, res_L, res_W, lr = self.sess.run(
                                [self.train_op, self.loss_op, self.res_L_op, self.res_W_op, self.lr_op],
                                feed_dict=self.construct_feed_dict(X_b, y_b=y_b, w_b=w_b))

                        if batch_num % 10 == 0:
                            print('loss = ' + str(loss_val))
                            # print('learning rate = {}'.format(str(lr)))

                            loss_curve.append(loss_val)
                            # self.train_writer.add_summary(loss_val, epoch)
                            # self.train_writer.add_summary(lr, epoch)
                            # if epoch == 10:
                            # self.watch_batch(X_b, ids_b, res_L, L)

                    if epoch % 5 == 0:
                        scores = self.evaluate(test_data, metrics)
                        for metric in metrics:
                            scores_curves[metric.name].append(scores[metric.name])
                            print("Metric {m} is {s}".format(m=metric.name, s=scores[metric.name]))

        return loss_curve, scores_curves

    # def predict(self, dataset, transformers=[], **kwargs):
    #     """Wraps predict to set batch_size/padding."""
    #     return super(SingletaskGraphClassifier, self).predict(
    #         dataset, transformers, batch_size=self.batch_size)
    #
    # def predict_proba(self, dataset, transformers=[], n_classes=2, **kwargs):
    #     """Wraps predict_proba to set batch_size/padding."""
    #     return super(SingletaskGraphClassifier, self).predict_proba(
    #         dataset, transformers, n_classes=self.n_classes, batch_size=self.batch_size)

    def predict_on_batch(self, X):
        """Return model output for the provided input.
        """
        if self.pad_batches:
            # make sure X size == batch size
            X = self.pad_graphs(self.batch_size, X)

        # run eval data through the model
        with self.sess.as_default():
            feed_dict = self.construct_feed_dict(X)
            # Shape (n_samples,)
            output = self.sess.run(self.outputs, feed_dict=feed_dict)
            batch_output = np.argmax(output, axis=1)
        return batch_output

    def predict_proba_on_batch(self, X, n_classes=2):
        """Returns class probabilities on batch"""
        # run eval data through the model
        if self.pad_batches:
            X = self.pad_graphs(self.batch_size, X)

        with self.sess.as_default():
            feed_dict = self.construct_feed_dict(X)
            batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

        return batch_outputs

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

    def evaluate(self, dataset, metrics, transformers=[]):
        """
        Evaluates the performance of this model on specified dataset.

        """
        if not isinstance(metrics, list):
            metrics = [metrics]

        y = dataset['y']  # 1-d label, just class id
        w = dataset['w']

        # make sure shape is (n-sample,)
        y = np.squeeze(y)
        w = np.squeeze(w)
        if not len(metrics):
            return {}
        else:
            mode = metrics[0].mode

        if mode == "classification":
            y_pred = self.predict_proba(dataset, self.n_classes)  # batch_size = None, return all
            # y_pred_print = self.model.predict(self.dataset, self.output_transformers).astype(int)
        else:
            y_pred = self.predict(dataset)
            # y_pred_print = y_pred

        scores = {}
        for metric in metrics:
            scores[metric.name] = metric.compute_singletask_metric(y, y_pred, w)

        return scores

    def predict(self, dataset):

        y_preds = []
        X = dataset['X']
        total_num_batch = int(X.shape[0] // self.batch_size)

        for batch_num in range(total_num_batch):
            begidx = batch_num * self.batch_size
            endidx = (batch_num + 1) * self.batch_size
            X_batch = X[begidx: endidx]
            n_samples = len(X_batch)

            y_pred_batch = self.predict_on_batch(X_batch)
            # Discard any padded predictions
            y_pred_batch = y_pred_batch[:n_samples]
            y_pred_batch = np.reshape(y_pred_batch, (n_samples,))
            y_preds.append(y_pred_batch)

        y_pred = np.concatenate(y_preds)

        # The iterbatches does padding with zero-weight examples on the last batch.
        # Remove padded examples.
        n_samples = len(X)
        y_pred = y_pred[:n_samples]
        y_pred = np.reshape(y_pred, (n_samples, ))

        return y_pred

    def predict_proba(self, dataset, n_classes=2):
        """
        TODO: Do transformers even make sense here?

        Returns:
          y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
        """
        X = dataset['X']
        total_num_batch = int(X.shape[0] // self.batch_size)
        y_preds = []
        for batch_num in range(total_num_batch):

            begidx = batch_num * self.batch_size
            endidx = (batch_num + 1) * self.batch_size
            X_batch = X[begidx: endidx]

            n_samples = len(X_batch)
            y_pred_batch = self.predict_proba_on_batch(X_batch)
            y_pred_batch = y_pred_batch[:n_samples, :]
            y_pred_batch = np.reshape(y_pred_batch, (n_samples, n_classes))
            y_preds.append(y_pred_batch)

        if len(X[total_num_batch * self.batch_size:]) > 0:
            """ after for there is some sample left, Or for loop is executed"""
            X_batch = X[total_num_batch * self.batch_size:]
            n_samples = len(X_batch)
            y_pred_batch = self.predict_proba_on_batch(X_batch)
            y_pred_batch = y_pred_batch[:n_samples, :]
            y_pred_batch = np.reshape(y_pred_batch, (n_samples, n_classes))
            y_preds.append(y_pred_batch)

        y_pred = np.concatenate(y_preds)
        n_samples = len(X)
        y_pred = y_pred[:n_samples]
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
        return y_pred

