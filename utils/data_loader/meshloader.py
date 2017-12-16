
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pickle
import os
import time

from os import listdir
from os.path import isfile, join
import sklearn.cluster as sc
import pickle

from AGCN.utils.data_loader import DataLoader, PointcloudLoader
from AGCN.models import Graph

from AGCN.utils.transformer import BalancingTransformer, NormalizationTransformer
from AGCN.utils.dataset import DiskDataset, TrainTestDataset, STDiskDataset
from AGCN.utils.splitter import IndexSplitter, ScaffoldSplitter, RandomSplitter
from AGCN.utils import provider as pd
from AGCN.utils.save import save_to_disk, load_from_disk

# ModelNet40 official train/test split
BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN/')
TRAIN_FILES = pd.getDataFiles( \
    os.path.join(BASE_DIR, 'data/3Dmesh/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = pd.getDataFiles(\
    os.path.join(BASE_DIR, 'data/3Dmesh/modelnet40_ply_hdf5_2048/test_files.txt'))
TRAIN_FILES = list(map(lambda x: x.split('/')[-1], TRAIN_FILES))
TEST_FILES = list(map(lambda x: x.split('/')[-1], TEST_FILES))

NUM_POINT = 1024

TRAIN_NUM = 6000
TEST_NUM = 1024

NUM_CLUSTER=50


class MeshLoader(PointcloudLoader):
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
        Returns: creatd Train and Test dataset
        """
        assert os.path.exists(self.processed_data_dir)
        assert os.path.exists(self.data_dir)

        train_dir = os.path.join(self.processed_data_dir, 'train')
        test_dir = os.path.join(self.processed_data_dir, 'test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        metadata_rows = []
        all_X, all_y, all_w, all_ids = [], [], [], []
        for shard_num, (shard, y) in enumerate(self.get_shards(TRAIN_FILES, shard_size)):
            """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
                y is the class id for each sample (3D mesh), int type
            """
            print('Featurizing Training Data , Shard - %d' % shard_num)
            X = self.featurize_shard(shard)
            ids = np.arange(X.shape[0]) + shard_num * shard_size  # ids is the null here. no use
            w = np.ones((y.shape[0],))
            assert len(X) == len(y) == len(w) == len(ids)

            """ save shard (x,y,ids, w)"""
            basename = "shard-%d" % shard_num
            metadata_rows.append(
                self.write_data_to_disk(train_dir, basename, X, y, w, ids))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            all_ids.append(ids)

        """ save the meta data to local """
        # metadata_df = STDiskDataset._construct_metadata(metadata_rows)
        # metadata_filename = os.path.join(train_dir, "metadata.joblib")
        # save_to_disk((self.tasks, metadata_df), metadata_filename)

        meta_data1 = list()
        meta_data1.append(metadata_rows)
        with open(os.path.join(train_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta_data1, f)

        """ return a Dataset contains all X, y, w, ids"""
        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_w = np.squeeze(np.concatenate(all_w))
        all_ids = np.squeeze(np.concatenate(all_ids))
        assert all_X.shape[0] == all_y.shape[0] == all_w.shape[0] == all_ids.shape[0]

        # create output dataset
        train_dataset = dict()
        train_dataset['X'] = all_X
        train_dataset['y'] = all_y
        train_dataset['w'] = all_w
        train_dataset['ids'] = all_ids


        metadata_rows = []
        all_X, all_y, all_w, all_ids = [], [], [], []
        for shard_num, (shard, y) in enumerate(self.get_shards(TEST_FILES, shard_size)):
            """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
                y is the class id for each sample (3D mesh), int type
            """
            print('Featurizing Testing Data , Shard - %d' % shard_num)
            X = self.featurize_shard(shard)
            ids = np.arange(X.shape[0]) + shard_num * shard_size  # ids is the null here. no use
            w = np.ones((y.shape[0],))
            assert len(X) == len(y) == len(w) == len(ids)

            """ save shard (x,y,ids, w)"""
            basename = "shard-%d" % shard_num
            metadata_rows.append(
                self.write_data_to_disk(test_dir, basename, X, y, w, ids))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            all_ids.append(ids)

        """ save the meta data to local """
        meta_data2 = list()
        meta_data2.append(metadata_rows)
        with open(os.path.join(test_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta_data2, f)

        """ return a Dataset contains all X, y, w, ids"""
        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_w = np.squeeze(np.concatenate(all_w))
        all_ids = np.squeeze(np.concatenate(all_ids))
        assert all_X.shape[0] == all_y.shape[0] == all_w.shape[0] == all_ids.shape[0]

        # create output dataset
        test_dataset = dict()
        test_dataset['X'] = all_X
        test_dataset['y'] = all_y
        test_dataset['w'] = all_w
        test_dataset['ids'] = all_ids

        return train_dataset, test_dataset

    @staticmethod
    def write_data_to_disk(
            data_dir,
            basename,
            X=None,
            y=None,
            w=None,
            ids=None):

        out_X = "%s-X.joblib" % basename
        save_to_disk(X, os.path.join(data_dir, out_X))

        out_y = "%s-y.joblib" % basename
        save_to_disk(y, os.path.join(data_dir, out_y))

        out_w = "%s-seg.joblib" % basename
        save_to_disk(w, os.path.join(data_dir, out_w))

        out_ids = "%s-y1h.joblib" % basename
        save_to_disk(ids, os.path.join(data_dir, out_ids))

        # note that this corresponds to the _construct_metadata column order
        return {'basename': basename, 'X': out_X, 'y': out_y, 'w': out_w, 'ids': out_ids}

    def get_shards(self, file_list, shard_size):

        """ shuffle the data files"""
        file_idxs = np.arange(0, len(file_list))
        np.random.shuffle(file_idxs)  # randomly extract data from files

        all_data, all_label = [], []
        for fn in file_idxs:
            # load the point cloud from the file, current_data -> data from one file
            current_data, current_label = pd.loadDataFile(os.path.join(self.data_dir, file_list[fn]))
            current_data = current_data[:, 0:NUM_POINT, :]
            current_data, current_label, _ = pd.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)  # make it (n-sample, ) shape
            all_data.append(current_data)
            all_label.append(current_label)
        """ create numpy for all data and label"""
        all_data = np.squeeze(np.vstack(all_data))
        all_label = np.squeeze(np.hstack(all_label))
        assert all_label.shape[0] == all_data.shape[0]

        shard_num = 0
        # pc_dir is the directory to point cloud files
        if shard_size is None:
            rotated_data = pd.rotate_point_cloud(all_data)
            jittered_data = pd.jitter_point_cloud(rotated_data)
            yield jittered_data, all_label
        else:
            start_idx = shard_num * shard_size
            end_idx = (shard_num + 1) * shard_size
            while end_idx < all_data.shape[0]:
                rotated_data = pd.rotate_point_cloud(all_data[start_idx:end_idx, :, :])
                jittered_data = pd.jitter_point_cloud(rotated_data)  # ready data
                sel_labels = all_label[start_idx:end_idx]       # corresponding labels
                shard_num += 1
                start_idx = shard_num * shard_size
                end_idx = (shard_num + 1) * shard_size
                yield jittered_data, sel_labels
            # end exceed the bound of pc_names, return all
            yield pd.jitter_point_cloud(pd.rotate_point_cloud(all_data[start_idx:, :, :])), all_label[start_idx:]

    def featurize_shard(self, shard):

        """
        convert ndarray (n-sample, n_point of sample, 3-d)
        :param shard: ndarray, point cloud raw data
        :return: graph object for each sample
        """
        cluster_num = NUM_CLUSTER  # min_node in this folder is 13 precomputed
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

    def load_back_from_disk(self, data_dir):
        """
        load data backas Train/test from disk
        :return: Train/Test STDiskDataset
        """
        """load back metadata_df"""
        meta_data = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
        metadata_rows = meta_data[0]

        """itershard by loading from disk"""
        all_X, all_y, all_w, all_ids = [], [], [], []

        for _, row in enumerate(metadata_rows):
            X = np.array(load_from_disk(os.path.join(data_dir, row['X'])))
            ids = np.array(load_from_disk(os.path.join(data_dir, row['ids'])))
            y = np.array(load_from_disk(os.path.join(data_dir, row['y'])))
            w = np.array(load_from_disk(os.path.join(data_dir, row['w'])))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            all_ids.append(ids)

        """ return a Dataset contains all X, y, w, ids"""

        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_w = np.squeeze(np.concatenate(all_w))
        all_ids = np.squeeze(np.concatenate(all_ids))
        assert all_X.shape[0] == all_y.shape[0] == all_w.shape[0] == all_ids.shape[0]

        # create output dataset
        dataset = dict()
        dataset['X'] = all_X
        dataset['y'] = all_y
        dataset['w'] = all_w
        dataset['ids'] = all_ids
        return dataset

    def load(self):
        """Load chemical datasets. Raw data is given as SMILES format"""

        meta_data = []
        if len(os.listdir(self.processed_data_dir)) != 0:

            print("Loading Saved Data from Disk.......")

            """ pre-defined location for saving the train and test data"""
            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            train = self.load_back_from_disk(data_dir=train_dir)
            test = self.load_back_from_disk(data_dir=test_dir)

            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            max_atom = meta_data[0]

        else:
            print("Loading and Featurizing Data.......")

            train, test = self.featurize(shard_size=256)

            """max_atom is the max atom of molecule in all_dataset """
            max_atom = NUM_CLUSTER
            meta_data.append(max_atom)
            meta_data.extend(self.tasks)
            with open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta_data, f)

        return self.tasks, (train, test), self.transformers, max_atom


