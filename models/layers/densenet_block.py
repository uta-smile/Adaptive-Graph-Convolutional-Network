from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import math

from AGCN.models.layers import Layer
from AGCN.models.layers.blockend import BlockEnd
from AGCN.models.operators import activations


class DenseBlockEnd(Layer):

    """
    Dense connected network on graph. This DenseBlockEnd layer add up all previous
    activations in block.
    Each block has 2 SGC_LL Layers
    """

    def __init__(self,
                 block_id,
                 res_n_features,
                 n_features,
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
        self.res_n_features = res_n_features
        self.n_features = n_features
        self.vars = {}  # save feature transform matrix to be learned

    def call(self, x):
        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice
        atom_features = x[:self.batch_size]
        mol_slice = x[self.batch_size]

        previous_activations = x[self.batch_size + 1:]
        assert len(previous_activations) % int(self.batch_size) == 0

        l_residuals = []    # contain activations within this block, and previous blocks' output
        for i in range(int(math.floor(len(previous_activations) / float(self.batch_size)))):
            l_residuals.append(previous_activations[self.batch_size * i: self.batch_size * (i+1)])
        # else:
        #     x_blocks = previous_activations[self.batch_size * (i+1):]
        # l_residuals += x_blocks
        "add saved last block graph atom features"
        atom_features_add_res = self.add_l_residuals(
            atom_features,
            mol_slice,
            l_residuals)

        # activation
        activated_atoms = map(lambda s: self.activation(s), atom_features_add_res)

        return activated_atoms

    def add_l_residuals(self, features, mol_slice, l_residuals):
        assert len(l_residuals) > 0
        assert len(features) == len(l_residuals[0])

        for layer_id in range(len(l_residuals)):
            residual_features = l_residuals[layer_id]
            x_add_res = []
            for graph_id in range(self.batch_size):
                x = features[graph_id]  # max_atom x Fin
                x_res = residual_features[graph_id]
                max_atom = self.max_atom

                x_indices = tf.gather(mol_slice, tf.pack([graph_id]))  # n_atom for this mol * feature number (,2) -> shape
                x = tf.slice(x, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
                x_res = tf.slice(x_res, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))
                x_new = tf.add(x, x_res)
                # [0, L.get_shape().as_list()[0] is max_atom
                M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
                x_new_pad = tf.pad(x_new, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
                x_add_res.append(x_new_pad)
            features = x_add_res
        return features
