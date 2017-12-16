from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from AGCN.models.operators import activations
from AGCN.models.layers import Layer, Dropout


def truncate_normal(shape, stddev=1e-3, name=None):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros init."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class SGC_LL(Layer):
    """
    SGC_LL -> SpecGraphConv with Learning Laplacian
    This SGC_LL layer is based on SpecGraphConv layer but it added the learning module for Laplacian matrix for each
    input graph in batch.
    Basically it gives the metric learning variable W to for dist(x,u) = sqrt((x-u)^T W W^T (x-u))
    W is added to trainable variable set and let it updated along with other parameters.
    """
    def __init__(self,
                 output_dim,
                 input_dim,
                 batch_size,
                 activation='relu',
                 dropout=None,
                 K=2,
                 save_lap=False,
                 save_output=False,
                 **kwargs):

        super(SGC_LL, self).__init__(**kwargs)
        self.dropout = dropout
        self.activation = activations.get(activation)

        self.batch_size = batch_size
        # self.sparse_inputs = True
        self.nb_filter = output_dim
        self.n_atom_feature = input_dim
        self.vars = {}
        self.bias = True
        self.K = K
        self.save_lap = save_lap    # if True, save the Laplacian matrix of this layer
        self.early_laps = None
        self.save_output = save_output  # if yes, save it to graph model and use later as residuals

    def build(self):
        """
        differnt from build in GraphCov here self.vars['bias'] \[weights_1,2,] represent the W_list and b_list
        in GraphConv
        :return:
        """
        with tf.variable_scope(self.name + '_vars'):
            # for i in range(self.batch_size):
            # initialization
            self.vars['weight'] = glorot([self.n_atom_feature * self.K, self.nb_filter], name='weights_feature')
            if self.bias:
                self.vars['bias'] = zeros([self.nb_filter], name='bias')
            """ NOTE: this M_L matrix is as S= M_L * M_L^T in generalized Mahalanobis distance
                This matrix transform the orginal data features of n_atom_features length
            """
            self.vars['M_L'] = glorot([self.n_atom_feature, self.n_atom_feature], name='Maha_dist')
            self.vars['alpha'] = tf.get_variable('alpha', (1,), initializer=tf.constant_initializer(1.0),
                                                 dtype=tf.float32)

    def call(self, x):
        """
        :param x: feed from Feed_dict [atoms_dict, deg_adj_dict] in function batch_to_feed_dict(self, batch)
         in graph_topology.py
        :return: output tensor
        """
        # Add trainable weights
        self.build()

        # Extract graph topology
        # here because we use GraphTopologyMol, feed_dict -> x
        # order: atom_feature(list, batch_size * 1), Laplacian(list, batch_size * 1), mol_slice, L_slice
        # atom_features = x[:self.batch_size]
        # Laplacian = x[self.batch_size: self.batch_size * 2]
        # mol_slice = x[-2]
        # L_slice = x[-1]

        node_features = x['node_features']
        Laplacian = x['original_laplacian']
        mol_slice = x['data_slice']
        L_slice = x['lap_slice']

        "spectral convolution and Laplacian update"
        node_features, L_updated, W_updated = self.specgraph_LL(
                                                        node_features,
                                                        Laplacian,
                                                        mol_slice,
                                                        L_slice,
                                                        self.vars,
                                                        self.K,
                                                        self.n_atom_feature)

        # activation
        activated_nodes = []
        for i in range(self.batch_size):
            activated_mol = self.activation(node_features[i])
            if self.dropout is not None:
                activated_mol = Dropout(self.dropout)(activated_mol)
            activated_nodes.append(activated_mol)

        return activated_nodes, L_updated, W_updated

    def specgraph_LL(self, node_features, Laplacian, mol_slice, L_slice, vars, K, Fin):
        """
        This function perform :1) learn Laplacian with updated Maha weight M_L 2) do chebyshev approx spectral convolution
        Args:
            atoms: list of features for two
            mol_slice: list of start, end index to extract features from tensor
            vars: trainable variable: w, b and M_L
            K: degree of chebyshev approx
            Fin: n_feat -> 81 for atom bond feature concatation

        Returns: list of atoms features without non-linear
        """
        batch_size = len(node_features)
        x_conved = []
        M_L = vars['M_L']
        alpha = vars['alpha']
        res_L_updated = []
        res_W_updated = []  # similarity matrix
        for mol_id in range(batch_size):
            x = node_features[mol_id]  # max_atom x Fin
            L = Laplacian[mol_id]  # max_atom x max_atom
            max_atom = L.get_shape().as_list()[0]  # dtype=int

            # TODO- x(L)_indices should be tensor made in bacth_ro_feed by [start index for all dimension] and [size for all dimension]
            x_indices = tf.gather(mol_slice, tf.constant([mol_id]))  # n_atom for this mol * feature number (,2) -> shape
            L_indices = tf.gather(L_slice, tf.constant([mol_id]))
            x = tf.slice(x, tf.constant([0, 0]), tf.reduce_sum(x_indices, axis=0))  # M x Fin, start=[0,0] size = [M, -1]
            LL = tf.slice(L, tf.constant([0, 0]), tf.reduce_sum(L_indices, axis=0))  # M x M

            # LL = tf.clip_by_average_norm(LL, 1.0)
            # M, Fin = x.get_shape().as_list()[0], x.get_shape().as_list()[1]
            # M, Fin = int(M), int(Fin)   # M-> n_atom for this mol, Fin-> num of feature of input graph

            # receive the tensor
            # L = Laplacian_tensor(x, vars['M_L'])

            def func(x, M):
                x_w = np.dot(x, M)
                # mean_xw = np.mean(x_w, axis=0)
                # dev_xw = x_w - mean_xw * x.shape[0]
                # var = np.var(np.linalg.norm(dev_xw, axis=0))
                D = [[] for _ in range(x_w.shape[0])]  # adjacency matrix
                for i in range(x_w.shape[0]):
                    for j in range(x_w.shape[0]):
                        if j == i:
                            D[i].append(0.0)
                            continue
                        u = x_w[i]
                        v = x_w[j]
                        dist = np.linalg.norm(u - v)
                        value = 1 * np.exp(-1 * dist)
                        D[i].append(value)
                """ W - similarity matrix """
                W = np.asarray(D).astype(np.float32)
                adj_m = W
                """normalized adjacency matrix """
                # W_temp = W + np.eye(W.shape[0])     # W_temp o get the normalized adj_m
                # rowsum = W_temp.sum(axis=0)
                # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                # d_mat_inv_sqrt = np.diag(d_inv_sqrt)
                # adj_m = W.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

                """sparsify the graph"""
                # threshold = np.mean(adj_m)
                # adj_m = np.where(adj_m > threshold, _, 0.0)

                """get the normalized Laplacian L = I - D(-0.5)^T A D(-0.5)"""
                d = W.sum(axis=0)  # degree for each atom
                d += np.spacing(np.array(0, W.dtype))
                d = 1 / np.sqrt(d)
                D = np.diag(d.squeeze())  # D^{-1/2}
                I = np.identity(d.size, dtype=W.dtype)
                L = I - D * W * D
                # L = D - adj_m
                # if mol_id == 0:
                #     # trace = go.Heatmap(z=L)
                #     # data = [trace]
                #     # py.iplot(data, filename='test_basic-heatmap')
                #     # a = np.random.random((16, 16))
                #     plt.imshow(L, cmap='hot', interpolation='nearest')
                #     plt.show()
                return L.astype(np.float32), adj_m.astype(np.float32)

            res_L, res_W = tf.py_func(func, [x, M_L], [tf.float32, tf.float32])  # additional Laplacian Matrix
            res_L = tf.clip_by_average_norm(res_L, 1.0)
            res_L = activations.relu(res_L, alpha=alpha)
            # res_L_updated.append(res_L)
            res_W_updated.append(res_W)
            L_all = tf.add(res_L, LL)  # final Lapalcian
            # L_all = tf.clip_by_norm(L_all, 1.0)
            # L_all = activations.relu(L_all, alpha=alpha)
            res_L_updated.append(res_L)

            # Transform to Chebyshev basis
            x0 = x
            x = tf.expand_dims(x0, 0)  # x-> 1 x M x Fin

            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin
                return tf.concat(0, [x, x_])  # K x M x Fin

            # Chebyshev recurrence T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            if K > 1:
                x1 = tf.matmul(L_all, x0)
                x = concat(x, x1)
            for k in range(2, K):
                x2 = 2 * tf.matmul(L_all, x1) - x0  # M x Fin
                x = concat(x, x2)
                x0, x1 = x1, x2

            M = tf.squeeze(tf.gather(tf.transpose(x_indices, perm=[1, 0]), 0))
            shape = tf.stack([K, M, Fin])
            shape2 = tf.stack([M, K * Fin])
            x = tf.reshape(x, shape)

            x = tf.transpose(x, perm=[1, 2, 0])  # x -> M x Fin x K
            x = tf.reshape(x, shape2)  # x-> M x (Fin*K)
            w = vars['weight']  # weight matrix to transform feature vector (Fin*K x Fout)
            b = vars['bias']
            x = tf.matmul(x, w) + b  # x -> M x Fout + Fout

            x = tf.pad(x, paddings=[[0, tf.subtract(tf.constant(max_atom), M)], [0, 0]],
                       mode="CONSTANT")  # [0, L.get_shape().as_list()[0] is max_atom
            x_conved.append(x)  # M x nb_filter(Fout)
        return x_conved, res_L_updated, res_W_updated


