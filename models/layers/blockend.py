from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.layers import Layer
from AGCN.models.operators import activations

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
            max_atom = self.max_atom

            x_indices = tf.gather(mol_slice, tf.pack([graph_id]))  # n_atom for this mol * feature number (,2) -> shape
            x = tf.slice(x, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
            x_res = tf.slice(x_res, tf.pack([0, 0]), tf.reduce_sum(x_indices, axis=0))
            x_new = tf.add(x, x_res)
            # [0, L.get_shape().as_list()[0] is max_atom
            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            x_new_pad = tf.pad(x_new, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
            x_add_res.append(x_new_pad)
        return x_add_res


class DenseBlockEnd(BlockEnd):

    """
    Dense connected network on graph. This DenseBlockEnd layer add up all previous
    activations in block.
    Each block has 2 SGC_LL Layers
    """

    def __init__(self,
                 activation='relu',
                 *args,
                 **kwargs):

        super(DenseBlockEnd, self).__init__(*args, **kwargs)
        self.activation = activations.get(activation)

    def call(self, x):
        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice
        atom_features = x[:self.batch_size]
        mol_slice = x[self.batch_size]

        previous_activations = x[self.batch_size + 1:]
        assert len(previous_activations) % int(self.batch_size) == 0

        l_residuals = []
        for i in range(int(len(previous_activations) / int(self.batch_size))):
            l_residuals.append(previous_activations[self.batch_size * i: self.batch_size * (i+1)])

        "add saved last block graph atom features"
        atom_features_add_res = self.add_l_residuals(
            atom_features,
            mol_slice,
            l_residuals)

        # activation
        activated_atoms = []
        for i in range(self.batch_size):
            activated_mol = self.activation(atom_features_add_res[i])

            activated_atoms.append(activated_mol)

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
