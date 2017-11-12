from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.operators import initializations, activations
from AGCN.models.layers import Layer
from AGCN.models.layers.graphconv import glorot, truncate_normal


class FCL(Layer):
    """
    fully connected layer,
    input is Tensor (batch_size, n_feat)
    output is Tensor (batch_size, n_classes)

    Maybe used as output layer of classification
    """
    def __init__(self,
                 batch_size,
                 output_dim,
                 input_dim,
                 activation="relu",
                 bias=True,
                 **kwargs):
        super(FCL, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.bias = bias
        self.input_shape = (self.batch_size, self.input_dim,)
        self.vars = {}
        self.build()

    def build(self):
        with tf.variable_scope(self.name + '_vars'):
            self.vars['W'] = truncate_normal((self.input_dim, self.output_dim), name='{}_weight'.format(self.name))
            if self.bias:
                self.vars['b'] = self.add_weight(
                    (self.output_dim,), initializer='zero', name='{}_bias'.format(self.name))

    def __call__(self, x):
        self.build()

        X = x['node_features']

        X = tf.matmul(X, self.vars['W'])
        if self.bias:
            X = tf.nn.bias_add(X, self.vars['b'])

        X = self.activation(X)
        X = tf.reshape(X, [self.batch_size, self.output_dim])
        return X
