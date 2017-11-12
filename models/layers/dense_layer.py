from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.operators import initializations, activations
from AGCN.models.layers import Layer


class DenseMol(Layer):
    def __init__(self,
                 output_dim,
                 input_dim,
                 init='glorot_uniform',
                 activation="relu",
                 bias=True,
                 **kwargs):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bias = bias

        self.input_shape = (self.input_dim,)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseMol, self).__init__(**kwargs)

    def __call__(self, x):

        self.W = self.add_weight(
            (self.input_dim, self.output_dim),
            initializer=self.init,
            name='{}_W'.format(self.name))
        self.b = self.add_weight(
            (self.output_dim,), initializer='zero', name='{}_b'.format(self.name))

        output = []
        X = x['node_features']
        for xx in X:
            # xx = model_ops.dot(xx, self.W)
            xx = tf.matmul(xx, self.W)
            if self.bias:
                xx += self.b
            output.append(xx)
        return output


class Dense(Layer):
    """Just your regular densely-connected NN layer.

    TODO(rbharath): Make this functional in deepchem

    Example:

    >>> import deepchem as dc
    >>> # as first layer in a sequential model:
    >>> model = dc.models.Sequential()
    >>> model.add(dc.nn.Input(shape=16))
    >>> model.add(dc.nn.Dense(32))
    >>> # now the model will take as input arrays of shape (*, 16)
    >>> # and output arrays of shape (*, 32)

    >>> # this is equivalent to the above:
    >>> model = dc.models.Sequential()
    >>> model.add(dc.nn.Input(shape=16))
    >>> model.add(dc.nn.Dense(32))

    Parameters
    ----------
    output_dim: int > 0.
    init: name of initialization function for the weights of the layer
    activation: name of activation function to use
      (see [activations](../activations.md)).
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: a(x) = x).
    W_regularizer: (eg. L1 or L2 regularization), applied to the main weights matrix.
    b_regularizer: instance of regularize applied to the bias.
    activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
      applied to the network output.
    bias: whether to include a bias
      (i.e. make the layer affine rather than linear).
    input_dim: dimensionality of the input (integer). This argument
      (or alternatively, the keyword argument `input_shape`)
      is required when using this layer as the first layer in a model.

    # Input shape
      nD tensor with shape: (nb_samples, ..., input_dim).
      The most common situation would be
      a 2D input with shape (nb_samples, input_dim).

    # Output shape
      nD tensor with shape: (nb_samples, ..., output_dim).
      For instance, for a 2D input with shape `(nb_samples, input_dim)`,
      the output would have shape `(nb_samples, output_dim)`.
    """

    def __init__(self,
                 output_dim,
                 input_dim,
                 init='glorot_uniform',
                 activation="relu",
                 bias=True,
                 **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.bias = bias

        input_shape = (self.input_dim,)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense, self).__init__(**kwargs)
        self.input_dim = input_dim

    def __call__(self, x):
        self.W = self.add_weight(
            (self.input_dim, self.output_dim),
            initializer=self.init,
            name='{}_W'.format(self.name))
        self.b = self.add_weight(
            (self.output_dim,), initializer='zero', name='{}_b'.format(self.name))

        output = model_ops.dot(x, self.W)
        if self.bias:
            output += self.b
        return output
