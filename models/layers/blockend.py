from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.layers import Layer


class BlockEnd(Layer):
    """
    BlockEnd layer is the last layer of one ResNet block.
    A ResNet block may contain multiple graphconv and graphpool layers.

    This layer does not have training parameters
    """

    def __init__(self,
                 max_atom,
                 batch_size,
                 **kwargs):
        super(BlockEnd, self).__init__(**kwargs)
        self.max_atom = max_atom
        self.batch_size = batch_size

    def call(self, x):
        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice
        atom_features = x[:self.batch_size]
        mol_slice = x[self.batch_size]
        residual_features = x[self.batch_size + 1:]

        "add saved last block graph atom features"
        atom_features_add_res = self.add_residuals(
                                            atom_features,
                                            mol_slice,
                                            residual_features)

        return atom_features_add_res

    def add_residuals(self, features, mol_slice, residual_features):
        assert len(features) == len(residual_features)
        x_add_res = []

        for graph_id in range(self.batch_size):
            x = features[graph_id]  # max_atom x Fin
            x_res = residual_features[graph_id]
            # max_atom = self.max_atom
            #
            # x_indices = tf.gather(mol_slice, tf.pack([graph_id]))  # n_atom for this mol * feature number (,2) -> shape
            # x = tf.slice(x, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
            # x_res = tf.slice(x_res, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))
            x_new = x + x_res
            # [0, L.get_shape().as_list()[0] is max_atom
            # M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            # x_new = tf.pad(x_new, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
            x_add_res.append(x_new)
        return x_add_res
