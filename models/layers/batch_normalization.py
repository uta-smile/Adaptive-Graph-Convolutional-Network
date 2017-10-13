from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.layers import Layer
from AGCN.models.operators import initializations, regularizers, activations


class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    epsilon: small float > 0. Fuzz parameter.
    mode: integer, 0, 1 or 2.
      - 0: feature-wise normalization.
          Each feature map in the input will
          be normalized separately. The axis on which
          to normalize is specified by the `axis` argument.
          During training we use per-batch statistics to normalize
          the data, and during testing we use running averages
          computed during the training phase.
      - 1: sample-wise normalization. This mode assumes a 2D input.
      - 2: feature-wise normalization, like mode 0, but
          using per-batch statistics to normalize the data during both
          testing and training.
    axis: integer, axis along which to normalize in mode 0. For instance,
      if your input tensor has shape (samples, channels, rows, cols),
      set axis to 1 to normalize per feature map (channels axis).
    momentum: momentum in the computation of the
      exponential average of the mean and standard deviation
      of the data, for feature-wise normalization.
    beta_init: name of initialization function for shift parameter, or
      alternatively, TensorFlow function to use for weights initialization.
    gamma_init: name of initialization function for scale parameter, or
      alternatively, TensorFlow function to use for weights initialization.
    gamma_regularizer: instance of WeightRegularizer
      (eg. L1 or L2 regularization), applied to the gamma vector.
    beta_regularizer: instance of WeightRegularizer,
      applied to the beta vector.

    Input shape:
    Arbitrary. Use the keyword argument input_shape
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

    Output shape:
    Same shape as input.

    References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 epsilon=1e-3,
                 mode=0,
                 axis=-1,
                 momentum=0.99,
                 beta_init='zero',
                 gamma_init='one',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 **kwargs):
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = activations.get(gamma_regularizer)
        self.beta_regularizer = activations.get(beta_regularizer)
        if self.mode == 0:
            self.uses_learning_phase = True
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = (input_shape[self.axis],)

        self.gamma = self.add_weight(
            shape,
            initializer=self.gamma_init,
            regularizer=self.gamma_regularizer,
            name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(
            shape,
            initializer=self.beta_init,
            regularizer=self.beta_regularizer,
            name='{}_beta'.format(self.name))
        # Not Trainable
        self.running_mean = self.add_weight(
            shape, initializer='zero', name='{}_running_mean'.format(self.name))
        # Not Trainable
        self.running_std = self.add_weight(
            shape, initializer='one', name='{}_running_std'.format(self.name))

    def call(self, x):
        if not isinstance(x, list):
            input_shape = model_ops.int_shape(x)
        else:
            x = x[0]
            input_shape = model_ops.int_shape(x)
        self.build(input_shape)
        if self.mode == 0 or self.mode == 2:

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            x_normed, mean, std = model_ops.normalize_batch_in_training(
                x, self.gamma, self.beta, reduction_axes, epsilon=self.epsilon)

            if self.mode == 0:
                self.add_update([
                    model_ops.moving_average_update(self.running_mean, mean,
                                                    self.momentum),
                    model_ops.moving_average_update(self.running_std, std,
                                                    self.momentum)
                ], x)

                if sorted(reduction_axes) == range(model_ops.get_ndim(x))[:-1]:
                    x_normed_running = tf.nn.batch_normalization(
                        x,
                        self.running_mean,
                        self.running_std,
                        self.beta,
                        self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = tf.reshape(self.running_mean,
                                                        broadcast_shape)
                    broadcast_running_std = tf.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = tf.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
                    x_normed_running = tf.batch_normalization(
                        x,
                        broadcast_running_mean,
                        broadcast_running_std,
                        broadcast_beta,
                        broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                x_normed = model_ops.in_train_phase(x_normed, x_normed_running)

        elif self.mode == 1:
            # sample-wise normalization
            m = model_ops.mean(x, axis=-1, keepdims=True)
            std = model_ops.sqrt(
                model_ops.var(x, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (x - m) / (std + self.epsilon)
            x_normed = self.gamma * x_normed + self.beta
        return x_normed
