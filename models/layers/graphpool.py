from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import numpy as np

from AGCN.models.layers import Layer


class GraphPoolMol(Layer):

    def __init__(self, batch_size, **kwargs):
        """
        Parameters
        ----------
        max_deg: int, optional
          Maximum degree of atoms in molecules.
        min_deg: int, optional
          Minimum degree of atoms in molecules.
        """

        super(GraphPoolMol, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.sparse_inputs = True

    def get_output_shape_for(self, input_shape):
        """Output tensor shape produced by this layer."""
        # Extract nodes
        atom_features_shape = input_shape[0]

        # assert (len(atom_fe/atures_shape) == 2,
        #         "GraphPool only takes 2 dimensional tensors")
        return atom_features_shape

    def call(self, x, mask=None):
        """Execute this layer on input tensors.
        """

        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice

        node_features = x['node_features']
        Laplacian = x['original_laplacian']
        mol_slice = x['data_slice']
        L_slice = x['lap_slice']

        # spectral pooling
        node_features = self.graph_pool_mol(node_features, Laplacian, mol_slice, L_slice)
        return node_features

    def graph_pool_mol(self, atoms, Laplacian, mol_slice, L_slice):
        """
            Parameters
            ----------
            atoms: list
                tf.Tensor of shape (n_atom_by_mol, n_feat)
            Laplacian: list
                of Laplacian tensor of shape (max_atom, max_atom) for each mol in batch

            mol_slice: tf.Tensor (batchsize, 2)
            L_slice: tf.Tensor (batchsize, 2)
            Returns
            -------
            List of tf.Tensor
              Of shape (atom_by_mol, n_feat)
        """
        batch_size = len(atoms)
        return_x = []
        for mol_id in range(batch_size):
            x = atoms[mol_id]
            L = Laplacian[mol_id]
            max_atom = L.get_shape().as_list()[0]  # dtype=int

            # x_indices = tf.gather(mol_slice, tf.pack([mol_id]))  # n_atom for this mol * feature number (,2) -> shape
            # L_indices = tf.gather(L_slice, tf.pack([mol_id]))  # get the shape info for laplacian
            # x = tf.slice(x, tf.pack([0, 0]), x_indices)  # n_atom * feature_n, begin=[0,0] size=[atom_n, -1]
            # L = tf.slice(L, tf.pack([0, 0]), L_indices)  # n_atom * n_atom
            # M, Fin = x.get_shape()  # x=> (M, Fin)
            # M, Fin = int(M), int(Fin)  # M-> n_atom for this mol, Fin-> num of feature of input graph

            x_indices = tf.gather(mol_slice, tf.constant([mol_id]))  # n_atom for this mol * feature number (,2) -> shape
            L_indices = tf.gather(L_slice, tf.constant([mol_id]))  # get the shape info for laplacian
            x = tf.slice(x, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
            L = tf.slice(L, tf.constant([0, 0]), tf.reduce_sum(L_indices, axis=0))  # M x M
            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))

            def func(x, L):
                # n_atom = x.get_shape().as_list()[0]     # number of atoms for this molecule
                acc_x = []
                for i, l in enumerate(list(L)):
                    idx = np.nonzero(l)  # get ith row of L retrieve the index of non-zeros
                    self_neighbor_atoms = x[idx]  # get those rows from x, including self and neighbors
                    if len(self_neighbor_atoms) != 0:
                        pooled_x = np.amax(self_neighbor_atoms, axis=0)  # reduce max along the column -> 1 x n_atom
                    else:
                        pooled_x = x[i]
                    acc_x.append(pooled_x)  # each atom's feature of this mol get max pooling with its neighbors
                acc_x = np.vstack(acc_x)  # make it tensor
                return acc_x

            acc_x = tf.py_func(func, [x, L], Tout=tf.float32)

            # pad the tensor back to original size before slicing
            acc_x = tf.pad(acc_x, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]], mode="CONSTANT")
            return_x.append(acc_x)
        return return_x  # return list of tensor
