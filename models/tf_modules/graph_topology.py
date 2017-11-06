from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import numpy as np


def merge_dicts(l):
    """Convenience function to merge list of dictionaries."""
    merged = {}
    for dict in l:
        merged = merge_two_dicts(merged, dict)
    return merged


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


class GraphTopologyMol(object):
    """Manages placeholders associated with batch of graphs and their topology"""

    def __init__(self, n_feat=75, batch_size=50, max_atom=128, name='topology_mol', max_deg=10, min_deg=0):
        """
        Note that batch size is not specified in a GraphTopology object. A batch
        of molecules must be combined into a disconnected graph and fed to topology
        directly to handle batches.

        Parameters
        ----------
        n_feat: int
          Number of features per atom.
        name: str, optional
          Name of this manager.
        max_deg: int, optional
          Maximum #bonds for atoms in molecules.
        min_deg: int, optional
          Minimum #bonds for atoms in molecules.
        batch_size: int,
          Size of input data batch; two batch -? data batch and Laplacian batch
        max_atom: int, optional
          maximum number of atom in the molecules in the datasets
        """

        # self.n_atoms = n_atoms
        self.n_feat = n_feat
        self.name = name
        self.max_deg = max_deg
        self.min_deg = min_deg

        self.max_atom = max_atom
        self.batch_size = batch_size
        # create placeholder receive the input data tensor!
        self.atom_features_placeholder = [
            tf.placeholder(
            dtype='float32',
            shape=(self.max_atom, self.n_feat),
            name=self.name + '_atom_features_' + str(i)) for i in range(batch_size)]

        self.data_slice_placeholder = tf.placeholder(
            dtype='int32',
            shape=(batch_size, 2),
            name=self.name + '_data_slice')     # for each mol, atom_feature (atom_n, feature_n)

        self.Laplacian_placeholder = [
            tf.placeholder(
            dtype='float32',
            shape=(self.max_atom, self.max_atom),
            name=self.name + '_laplacian_' + str(i)) for i in range(batch_size)]

        self.L_slice_placeholder = tf.placeholder(
            dtype='int32',
            shape=(batch_size, 2),              # for each mol, atom
            name=self.name + '_L_slice')

        self.topology = self.Laplacian_placeholder + [self.data_slice_placeholder, self.L_slice_placeholder]
        self.inputs = self.atom_features_placeholder + self.topology
        # self.inputs += self.topology

    def pad_data2sparse(self, graph):
        # get the feature and original shape
        feature, shape = graph.node_features, graph.node_features.shape
        pad_shape = (0, self.max_atom - shape[0])  # (0, number or row to pad)
        # feature : M_0 * F[0], shape:
        feature_pad = np.pad(feature, pad_width=(pad_shape, (0, 0)), mode='constant', constant_values=[0.])
        return feature_pad, np.asarray([shape[0], -1], dtype=np.int32)  # latter is size for rf.slice(x, begin, size) shape[0]-> atom_n, -1 -> all in dim2

    def pad_Lap2sparse(self, graph):
        # L -> ndarray, L_pad -> ndarray
        # pad the Laplacian from shape (atom_n, atom_n) -> shape (max_atom, max_atom)
        Laplacian, L_shape = graph.Laplacian, graph.Laplacian.shape
        pad_shape = ((0, self.max_atom - L_shape[0]), (0, self.max_atom - L_shape[1]))  # (0, number or row to pad)
        L_pad = np.pad(Laplacian.todense(), pad_width=pad_shape, mode='constant', constant_values=[0.])
        return L_pad, np.asarray(Laplacian.shape, dtype=np.int32)

    def batch_to_feed_dict(self, batch):
        """Converts the current batch of mol_graphs into tensorflow feed_dict.
        Assign the graph information to the placeholder made for each molecules in batch
        Args:
            batch: ndarray of Graph objects represents molecules
        Returns:
            return list of feed dict for each graphs in batch
        """
        mol_atoms = []      # save atom feature
        mol_slice = []      # save atom feature shape
        Laplacians = []
        L_slice = []

        for idx, graph in enumerate(list(batch)):
            data, shape = self.pad_data2sparse(graph)
            mol_atoms.append(data)
            mol_slice.append(shape)
        mol_slice = np.asarray(mol_slice, dtype=np.int32)
        # Generate data dicts
        atoms_dict = dict(
            list(zip(self.atom_features_placeholder, mol_atoms)))

        mol_slice_dict = {self.data_slice_placeholder: mol_slice}

        for idx, graph in enumerate(list(batch)):
            L, shape = self.pad_Lap2sparse(graph)
            Laplacians.append(L)
            L_slice.append(shape)
        L_slice = np.asarray(L_slice, dtype=np.int32)

        laplacian_dict = dict(
            list(zip(self.Laplacian_placeholder, Laplacians)))

        L_slice_dict = {self.L_slice_placeholder: L_slice}

        return merge_dicts([atoms_dict, laplacian_dict, mol_slice_dict, L_slice_dict])

    def get_input_placeholders(self):
        """All placeholders.

        Contains atom_features placeholder and topology placeholders.
        """
        return self.inputs

    def get_topology_placeholders(self):
        """Returns topology placeholders

        Consists of deg_slice_placeholder, membership_placeholder, and the
        deg_adj_list_placeholders.
        """
        return self.topology

    def get_datas_placeholders(self):
        return self.atom_features_placeholder + [self.data_slice_placeholder]

    def get_dataslice_placeholders(self):
        return self.data_slice_placeholder

    def get_atom_features_placeholder(self):
        return self.atom_features_placeholder

    def get_laplacians_placeholder(self):
        return self.Laplacian_placeholder

    def get_lapslice_placeholders(self):
        return self.L_slice_placeholder

