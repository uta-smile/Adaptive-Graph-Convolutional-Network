"""
Data structure for graph object used for training of AGCN
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import numpy as np
import networkx as nx
import scipy.sparse as sp


__author__ = "Ruoyu Li"
__copyright__ = "Copyright 2017 UT Arlington"


class Graph(object):
    """ Adaptive graph object for each data sample  """

    def __init__(self,
                 node_features,
                 adj_lists,
                 max_deg,
                 min_deg):
        """
        Parameters
        ----------
        node_features: np.ndarray
            shape (n_node, n_feat)
        adj_list: list
            List of length n_atoms, with neighbor indices of each atom.
        max_deg: int, optional
            Maximum degree of any node in graph.
        min_deg: int, optional
            Minimum degree of any node in graph.
        """

        self.node_features = node_features
        self.n_node, self.n_feat = node_features.shape
        # degree of each node
        self.degree_list = np.array([len(nbrs) for nbrs in adj_lists], dtype=np.int32)
        self.adj_lists = adj_lists
        self.max_deg = max_deg
        self.min_deg = min_deg

        self.has_Lap = True  # if has Laplacian, some molecule is too small to have a Laplacian matrix

        """initial sparse graph Laplacian: original graph (clustered or intrinsic)"""
        if self.n_node > 3:
            self.Laplacian = self.compute_laplacian(adj_lists)
        else:
            self.Laplacian = None
            self.has_Lap = False

    def get_num_nodes(self):
        # get the number of nodes in graph
        return self.n_node

    def get_num_features(self):
        # get the number of features in graph
        return self.n_feat

    def get_max_degree(self):
        return self.max_deg

    def get_original_adj_list(self):
        # return the adjacency list of original graph
        return self.adj_lists

    def get_node_fearture(self):
        return self.node_features

    @staticmethod
    def compute_adj_matrix(adj_lists):
        """convert adj_list of graph to adjacency matrix"""

        neighbors = {}  # atom_id : list of neighbor atom
        for index, list in enumerate(adj_lists):
            neighbors[index] = list  # atom i and its adj list
        adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(neighbors))
        return adj_matrix

    def compute_laplacian(self, adj_lists):

        def laplacian(adj_m, normalized=True):
            """Return the Laplacian of the weigth matrix.
            input: adjacency matrix (usually normalized)
            """
            W = adj_m
            # Degree matrix.
            d = W.sum(axis=0)

            # Laplacian matrix.
            if not normalized:
                D = sp.diags(d.A.squeeze(), 0)
                L = D - W
            else:
                d += np.spacing(np.array(0, W.dtype))
                d = 1 / np.sqrt(d)
                D = sp.diags(d.A.squeeze(), 0)
                I = sp.identity(d.size, dtype=W.dtype)
                L = I - D * W * D

            assert np.abs(L - L.T).mean() < 1e-9
            assert type(L) is sp.csr_matrix
            return L

        def normalize_adj(adj_m):
            """Symmetrically normalize adjacency matrix.
            adj_m -> dense adjacency matrix of graph
            """
            adj = sp.coo_matrix(adj_m)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return normalized_adj

        def preprocess_adj(adj_m):
            """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
            normalized_adj_m = normalize_adj(adj_m + sp.eye(adj_m.shape[0]))
            return normalized_adj_m  # return sparse matrix -> normalized  adj_matrix

        adj_m = self.compute_adj_matrix(adj_lists).astype(np.float32)
        adj_m = preprocess_adj(adj_m)
        lap_m = laplacian(adj_m)  # normalized is True, will give normalized Laplacian matrix
        return lap_m


class MolGraph(Graph):
    """
    for molecular data, whose graph derived from SMILES string
    """
    def __init__(self,
                 node_features,
                 adj_lists,
                 max_deg=10,
                 min_deg=0,
                 smiles=None):
        super(MolGraph, self).__init__(node_features, adj_lists, max_deg, min_deg)

        # record the smiles string for this molecule
        self.smiles = smiles




