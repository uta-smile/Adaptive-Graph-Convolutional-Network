"""
MultiLayer Perceptron (MLP), enrich the features of graph
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.operators import initializations, activations
from AGCN.models.layers import Layer
from AGCN.models.layers.graphconv import glorot


class MLP(Layer):
    def __init__(self,
                 output_dim,
                 hidden_dims,
                 input_dim,
                 batch_size,
                 init='glorot_uniform',
                 activation="relu",
                 bias=True,
                 max_atom=128,
                 **kwargs):

        super(MLP, self).__init__(**kwargs)
        assert type(hidden_dims) == list
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.bias = bias
        self.max_atom = max_atom
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.vars = {}

    def build(self):
        with tf.variable_scope(self.name + '_vars'):
            """ MLP weights W"""
            self.vars['weight'] = []
            self.vars['bias'] = []
            dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
            for id, l_dim in enumerate(zip(dims[:-1], dims[1:])):
                self.vars['weight'].append(glorot([l_dim[0], l_dim[1]], name='trans_feature_{}'.format(id)))

            for id, l_dim in enumerate(zip(dims[:-1], dims[1:])):
                self.vars['bias'].append(
                    tf.get_variable('bias_{}'.format(id), (l_dim[1],), initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
                )

    def call(self, x):

        self.build()

        node_features = x['node_features']
        mol_slice = x['data_slice']
        node_features = self.MLP(
            node_features,
            mol_slice,
        )
        return node_features

    def MLP(self, node_features, mol_slice):
        new_x = []
        for mol_id in range(self.batch_size):
            x = node_features[mol_id]  # max_atom x Fin
            x_indices = tf.gather(mol_slice, tf.constant([mol_id]))  # n_atom for this mol * feature number (,2) -> shape
            x = tf.slice(x, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]

            for op_id in range(len(self.vars['weight'])):
                x = tf.matmul(x, self.vars['weight'][op_id]) + self.vars['bias'][op_id]

            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            x = tf.pad(x, paddings=[[0, tf.subtract(tf.constant(self.max_atom), M)], [0, 0]],
                       mode="CONSTANT")
            new_x.append(tf.nn.relu(x))
        return new_x
