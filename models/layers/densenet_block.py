from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import math

from AGCN.models.layers import Layer
# from AGCN.models.layers.blockend import BlockEnd
from AGCN.models.operators import activations
from AGCN.models.layers.graphconv import glorot


class DenseBlockEnd(Layer):

    """
    Dense connected network on graph. This DenseBlockEnd layer add up all previous
    activations in block.
    Each block has 2 SGC_LL Layers
    """

    def __init__(self,
                 block_id,
                 res_n_features_list,
                 output_n_features,
                 activation='relu',
                 K=2,
                 max_atom=128,
                 batch_size=256,
                 **kwargs):

        super(DenseBlockEnd, self).__init__(**kwargs)
        self.max_atom = max_atom
        self.batch_size = batch_size
        self.block_id = block_id  # this id is to differentiate blocks
        self.activation = activations.get(activation)
        self.K = K
        assert type(res_n_features_list) == list
        self.res_n_features_list = res_n_features_list
        self.output_n_features = output_n_features
        self.vars = {}  # save feature transform matrix to be learned
        self.inblock_activations = []
        self.inblock_activations_dim = []
        self.preceding_blocks = []
        self.preceding_blocks_dim = []

    def build(self):
        """ build transform weight for feature from previous layers in block """
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight_inblock'] = []
            self.vars['weight_outblock'] = []
            for id, res_n_features in enumerate(self.preceding_blocks_dim):
                self.vars['weight_outblock'].append(
                    glorot([res_n_features, self.output_n_features], name='trans_outblock_feature_{}'.format(id))
                )

            for id, res_n_features in enumerate(self.inblock_activations_dim):
                self.vars['weight_inblock'].append(
                    glorot([res_n_features, self.output_n_features], name='trans_inblock_feature_{}'.format(id))
                )

            # weight on activations from previous layer
            self.vars['beta_inblock'] = tf.get_variable('beta1', (1,), initializer=tf.constant_initializer(1.0),
                                                        dtype=tf.float32)

            self.vars['beta_outblock'] = tf.get_variable('beta2', (1,), initializer=tf.constant_initializer(1.0),
                                                         dtype=tf.float32)

    def call(self, x):
        # atom_features = x[:self.batch_size]
        # mol_slice = x[self.batch_size]
        #
        # inblock_activations_dim = x[-2]
        # block_outputs_dim = x[-1]
        # assert len(previous_activations) % int(self.batch_size) == 0

        node_features = x['node_features']
        mol_slice = x['data_slice']
        self.inblock_activations = x['inblock_activations']
        self.inblock_activations_dim = x['inblock_activations_dim']
        self.preceding_blocks = x['block_outputs']
        self.preceding_blocks_dim = x['block_outputs_dim']

        self.build()

        "add saved last block graph atom features"
        node_features = self.add_l_residuals(
            node_features,
            mol_slice,
        )

        # activation
        activated_nodes = list(map(lambda s: self.activation(s), node_features))

        return activated_nodes

    def add_l_residuals(self, features, mol_slice):

        # for layer_id, _ in enumerate(self.preceding_blocks_dim):
        #     residual_features = self.preceding_blocks[layer_id]
        #     x_add_res = []
        beta1 = self.vars['beta_inblock']
        beta2 = self.vars['beta_outblock']
        x_add_res = []
        assert len(self.preceding_blocks) == len(self.preceding_blocks_dim) * self.batch_size
        for graph_id in range(self.batch_size):
            x = features[graph_id]  # max_atom x Fin
            # x_res = residual_features[graph_id]
            max_atom = self.max_atom
            "x_indices shared by this graph (including from previous layers)"
            x_indices = tf.gather(mol_slice, tf.constant([graph_id]))  # n_atom for this mol * feature number (,2) -> shape
            x = tf.slice(x, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]

            # sum up in-block activations
            for layer_id in range(len(self.inblock_activations_dim)):
                x_res = self.inblock_activations[layer_id * self.batch_size + graph_id]
                x_res = tf.slice(x_res, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))
                x += tf.multiply(tf.matmul(x_res, self.vars['weight_inblock'][layer_id]), beta1)

            # sum up out-of-block activations
            for b_id in range(len(self.preceding_blocks_dim)):
                x_res = self.preceding_blocks[b_id * self.batch_size + graph_id]
                x_res = tf.slice(x_res, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))
                x += tf.multiply(tf.matmul(x_res, self.vars['weight_outblock'][b_id]), beta2)

            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            x_new_pad = tf.pad(x, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
            x_add_res.append(x_new_pad)

        return x_add_res
