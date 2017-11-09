
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import sklearn.cluster as sc

from AGCN.utils.data_loader import DataLoader
from AGCN.utils.datatset import STDiskDataset
from AGCN.models import Graph

from AGCN.utils.transformer import BalancingTransformer, NormalizationTransformer
from AGCN.utils.datatset import DiskDataset
from AGCN.utils.splitter import IndexSplitter, ScaffoldSplitter, RandomSplitter
from AGCN.utils import provider as pd


# ModelNet40 official train/test split
BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN/')
TRAIN_FILES = pd.getDataFiles( \
    os.path.join(BASE_DIR, 'data/3Dmesh/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = pd.getDataFiles(\
    os.path.join(BASE_DIR, 'data/3Dmesh/modelnet40_ply_hdf5_2048/test_files.txt'))
TRAIN_FILES = map(lambda x: x.split('/')[-1], TRAIN_FILES)
TEST_FILES = map(lambda x: x.split('/')[-1], TEST_FILES)

NUM_POINT = 1024


class MeshLoader(DataLoader):
    """
    This class of loader is for Point Cloud data, which is usually lots of binary files stay in
    a folder. Each binary file is a point cloud object, reading it, you get a nddarray of 3D points, each rwo of which
    is a point in laser receivers.
    Input: folder directory
    Output: ConvMol instance for each point cloud object. Features is intensity, adjacency matrix, degree
    are manually learned and set by clustering.
    """

    def featurize(self, shard_size=1024):
        """
        Args:
            input_folder: folder name of binary files (graph object)
            data_dir: where to save generated data frame
            shard_size: number of graphs in one shard
        Returns:
        """
        data_dir = self.processed_data_dir  # where to store the processed sharded data
        assert os.path.exists(data_dir)
        assert os.path.exists(self.data_dir)

        def shard_generator():
            for shard_num, (shard, y) in enumerate(self.get_shards(TRAIN_FILES, shard_size)):
                """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
                    y is the class id for each sample (3D mesh), int type
                """
                X = self.featurize_shard(shard)
                ids = np.arange(X.shape[0])  # ids is the null here. no use
                w = np.ones((y.shape[0], ))
                assert len(X) == len(y) == len(w)
                yield X, y, w, ids

        return STDiskDataset.create_dataset(
            shard_generator(), data_dir, self.tasks, verbose=self.verbose)

    def get_shards(self, file_list, shard_size):

        train_file_idxs = np.arange(0, len(file_list))
        np.random.shuffle(train_file_idxs)        # randomly extract data from files

        shard_num = 0
        for fn in range(len(file_list)):
            # load the point cloud from the file
            current_data, current_label = pd.loadDataFile(os.path.join(self.data_dir, file_list[train_file_idxs[fn]]))
            current_data = current_data[:, 0:NUM_POINT, :]
            current_data, current_label, _ = pd.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)   # make it (n-sample, ) shape

            # pc_dir is the directory to point cloud files
            if shard_size is None:
                rotated_data = pd.rotate_point_cloud(current_data)
                jittered_data = pd.jitter_point_cloud(rotated_data)
                yield (jittered_data, current_label)
            else:
                start_idx = shard_num * shard_size
                end_idx = (shard_num + 1) * shard_size
                while end_idx <= current_data.shape[0]:
                    rotated_data = pd.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                    jittered_data = pd.jitter_point_cloud(rotated_data)  # ready data
                    sel_labels = current_label[start_idx:end_idx]       # corresponding labels
                    shard_num += 1
                    start_idx = shard_num * shard_size
                    end_idx = (shard_num + 1) * shard_size
                    yield (jittered_data, sel_labels)
                # end exceed the bound of pc_names, return all
                yield (pd.jitter_point_cloud(pd.rotate_point_cloud(current_data[start_idx:, :, :])), current_label[start_idx:])

    def featurize_shard(self, shard):

        """
        convert ndarray (n-sample, n_point of sample, 3-d)
        :param shard: ndarray, point cloud raw data
        :return: graph object for each sample
        """
        cluster_num = 50  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []  # to save the graph object
        n_samples = shard.shape[0]
        for pc_id in range(n_samples):
            # iterate each pc in shard
            P = np.squeeze(shard[pc_id, :, :])
            if P.shape[0] > cluster_num:
                # clustering on original cloud
                cluster_indicators = clusterer.fit_predict(P)  # labels of original points
                new_points = []
                for i in range(cluster_num):
                    """ each cluster center is the new point after clustering"""
                    idx = np.where(cluster_indicators == i)
                    coord = np.mean(P[idx], axis=0)
                    new_points.append(coord)
                new_PC = np.asarray(new_points).astype(np.float32)
                assert new_PC.shape[0] <= cluster_num
            else:
                # do not need a clustering, use original points
                new_PC = P

            # create graph on point cloud
            adj_list, adj_matrix = self.get_adjacency(new_PC)
            X.append(Graph(new_PC, adj_list, max_deg=cluster_num, min_deg=0))

        return np.asarray(X)

    @staticmethod
    def get_adjacency(points):
        n_p = points.shape[0]

        # pre-learn the distance matrix, use mean as threshold to generate adjacency matrix, i.e. who connect who
        sum_dist = []
        for i in range(n_p):
            for j in range(i + 1):
                sum_dist += [np.linalg.norm(points[i] - points[j])]
                # input points are new points after clustering
        d_lim = np.mean(sum_dist)

        adj_matrix = np.zeros((n_p, n_p))
        adj_list = [[] for _ in range(n_p)]
        for i in range(n_p):
            for j in range(i + 1, n_p):
                if np.linalg.norm(points[i] - points[j]) < d_lim:
                    adj_matrix[i][j] = True
                    # adjacency list has redundancy
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        return adj_list, adj_matrix

    def load(self):
        """Load chemical datasets. Raw data is given as SMILES format"""

        meta_data = []
        if len(os.listdir(self.processed_data_dir)) != 0:

            print("Loading Saved Data from Disk.......")

            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            dataset = STDiskDataset(data_dir=self.processed_data_dir)
            train = STDiskDataset(data_dir=train_dir)
            test = STDiskDataset(data_dir=test_dir)

            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            max_atom = meta_data[0]

            print("Transforming Data.")
            if not self.transformer_types:
                if self.transformer_types == 'normalization_y':
                    self.transformers += [
                        NormalizationTransformer(transform_y=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'normalization_w':
                    self.transformers += [
                        NormalizationTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_w':
                    self.transformers += [
                        BalancingTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_y':
                    self.transformers += [
                        BalancingTransformer(transform_y=True, dataset=dataset)
                    ]
                else:
                    ValueError("Transformer type Not defined!{}".format(self.transformer_types))

        else:
            print("Loading and Featurizing Data.......")
            # loader = dc.data.CSVLoader(
            #     tasks=self.tasks, smiles_field=self.smiles_field, featurizer=self.Featurizer)
            dataset = self.featurize(shard_size=2048)

            print("Transforming Data.")
            if not self.transformer_types:
                if self.transformer_types == 'normalization_y':
                    self.transformers += [
                        NormalizationTransformer(transform_y=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'normalization_w':
                    self.transformers += [
                        NormalizationTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_w':
                    self.transformers += [
                        BalancingTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_y':
                    self.transformers += [
                        BalancingTransformer(transform_y=True, dataset=dataset)
                    ]
                else:
                    ValueError("Transformer type Not defined!{}".format(self.transformer_types))

            if len(self.transformers) > 0:
                for transformer in self.transformers:
                    # pass dataset through maybe more than one transformer
                    dataset = transformer.transform(dataset)

            """max_atom is the max atom of molecule in all_dataset """
            max_atom = self.find_max_atom(dataset)
            meta_data.append(max_atom)
            meta_data.extend(self.tasks)
            with open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta_data, f)

            """
            Split Dataset
            """
            print("Splitting Date to Train/Validation/Testing")
            splitters = {
                'index': IndexSplitter(),
                'random': RandomSplitter(),
                'scaffold': ScaffoldSplitter()
            }

            if self.splitter not in splitters:
                raise ValueError("Splitter not defined!")
            else:
                splitter = splitters[self.splitter]

            # create processed dirs as train, valid, test
            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            print("Saving Data at %s...", self.processed_data_dir)
            train, test = splitter.train_test_split(
                dataset,
                train_dir=train_dir,
                test_dir=test_dir,
                frac_train=self.split_frac[0],
            )

        return self.tasks, (train, test), self.transformers, max_atom


