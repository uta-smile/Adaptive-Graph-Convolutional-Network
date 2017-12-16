from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

from AGCN.models.operators import activations
from AGCN.models.layers import Layer


class GraphGatherMol(Layer):

    def __init__(self, batch_size, activation='relu', **kwargs):
        """
        Parameters
        ----------
        batch_size: int
          Number of elements in batch of data.
        """
        super(GraphGatherMol, self).__init__(**kwargs)

        self.activation = activations.get(activation)  # Get activations
        self.batch_size = batch_size

    def build(self, input_shape):
        """Nothing needed (no learnable weights)."""
        pass

    def get_output_shape_for(self, input_shape):
        """Output tensor shape produced by this layer."""
        # Extract nodes and membership
        atom_features_shape = input_shape[0]
        # membership_shape = input_shape[2]

        # assert (len(atom_features_shape) == 2,
        #         "GraphGather only takes 2 dimensional tensors")
        n_feat = atom_features_shape[1]
        return self.batch_size, n_feat

    def call(self, x, mask=None):

        # extract tensors or list of tensors
        node_features = x['node_features']
        mol_slice = x['data_slice']

        # Perform the mol gather
        node_features = self.graph_gather_mol(node_features, mol_slice, self.batch_size)
        return self.activation(node_features)

    def graph_gather_mol(self, atoms, mol_slice, batch_size):
        """
        Parameters
        ----------
        atoms: list of tf.Tensor
          Of shape (n_atoms, n_feat)
        Laplacian: list
            of Laplacian tensor of shape (max_atom, max_atom) for each mol in batch
        mol_slice: tf.Tensor (batch_size, 2)
        L_slice: tf.Tensor (batch_size, 2)

        Returns
        -------
        tf.Tensor
          Of shape (batch_size, n_feat)
        """
        # assert (batch_size > 1, "graph_gather requires batches larger than 1")  # cannot work with batch_size == 1
        mol_feature = []
        for mol_id in range(batch_size):
            x = atoms[mol_id]
            x_indices = tf.gather(mol_slice, tf.constant([mol_id]))  # n_atom for this mol * feature number (,2) -> shape
            # x = tf.slice(x, tf.constant([0, 0]), x_indices)  # n_atom * feature_n, start=[0,0] size = [atom_n, -1]
            x = tf.slice(x, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))

            f_mol = tf.reduce_sum(x, 0)     # along row axis sum
            mol_feature.append(f_mol)

        mol_reps = tf.stack(mol_feature)    # stack all mol_feature to a tensor -> (batch_size, feature_length)
        return mol_reps