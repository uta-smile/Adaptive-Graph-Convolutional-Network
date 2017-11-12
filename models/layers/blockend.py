from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.layers import Layer
from AGCN.models.operators import activations
from AGCN.models.layers.graphconv import glorot, truncate_normal


class BlockEnd(Layer):
    """
    BlockEnd layer is the last layer of one Residual graph Network block.
    A ResNet block may contain multiple graphconv and graphpool layers.

    res_n_features -> feature number of previous layer, (residual values)
    n_feature -> feature number of current layer

    """

    def __init__(self,
                 block_id,
                 res_n_features,
                 n_features,
                 activation='relu',
                 max_atom=128,
                 batch_size=256,
                 **kwargs):
        super(BlockEnd, self).__init__(**kwargs)
        self.max_atom = max_atom
        self.batch_size = batch_size
        self.block_id = block_id    # this id is to differentiate blocks
        self.activation = activations.get(activation)
        self.res_n_features = res_n_features
        self.n_features = n_features
        self.vars = {}  # save feature transform matrix to be learned

    def build(self):
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weight'] = glorot([self.res_n_features, self.n_features], name='trans_feature')

    def call(self, x):
        # Add trainable weights
        self.build()

        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice
        node_features = x['node_features']
        mol_slice = x['data_slice']
        residual_features = x['block_outputs']  # residual activations from last block

        "add saved last block graph atom features"
        node_features = self.add_residuals(
                                            node_features,
                                            mol_slice,
                                            residual_features,
                                            self.vars)

        # activation
        activated_nodes = map(lambda s: self.activation(s), node_features)

        return activated_nodes

    def add_residuals(self, node_features, mol_slice, residual_features, vars):
        assert len(node_features) == len(residual_features)
        x_add_res = []
        w = vars['weight']  # transform feature dimension between x_res and x

        for graph_id in range(self.batch_size):
            x = node_features[graph_id]  # max_atom x Fin
            x_res = residual_features[graph_id]
            max_atom = self.max_atom

            x_indices = tf.gather(mol_slice, tf.pack([graph_id]))  # n_atom for this mol * feature number (,2) -> shape
            x = tf.slice(x, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
            x_res = tf.slice(x_res, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))
            x_new = tf.add(tf.matmul(x_res, w), x)
            # [0, L.get_shape().as_list()[0] is max_atom
            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            x_new_pad = tf.pad(x_new, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
            x_add_res.append(x_new_pad)

        return x_add_res




