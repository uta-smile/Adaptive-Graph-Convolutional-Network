
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import os

from os import listdir
from os.path import isfile, join
import sklearn.cluster as sc
import pickle

from AGCN.utils.data_loader import PointcloudLoader
from AGCN.models import Graph

from AGCN.utils import provider as pd
from AGCN.utils.save import save_to_disk, load_from_disk


# ShapeNet Part Segmentation
BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN/data/3Dmesh/part_seg_shapenet/hdf5_data')
TRAIN_FILES = pd.getDataFiles(os.path.join(BASE_DIR, 'train_hdf5_file_list.txt'))
TEST_FILES = pd.getDataFiles(os.path.join(BASE_DIR, 'test_hdf5_file_list.txt'))

NUM_CATEGORIES = 16
CLUSTER_NUM = 100

TRAIN_NUM = 6000
TEST_NUM = 512


class ShapeNetLoader(PointcloudLoader):
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
        all_X, all_y, all_seg, all_y_one_hot = [], [], [], []
        for shard_num, (shard, y, seg, y_1h) in enumerate(self.get_shards(TRAIN_FILES, shard_size)):
            """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
                y is the class id for each sample (3D mesh), int type
            """
            print('Featurizing Training Data , Shard - %d' % shard_num)
            X, seg = self.featurize_shard(shard, seg)

            """ save shard (x,y,ids, w)"""
            basename = "shard-%d" % shard_num
            metadata_rows.append(self.write_data_to_disk(train_dir, basename, X, y, seg, y_1h))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_seg.append(seg)
            all_y_one_hot.append(y_1h)

        """ save the meta data of training dataset to local """
        meta_data1 = list()
        meta_data1.append(metadata_rows)
        with open(os.path.join(train_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta_data1, f)

        """ return a Dataset contains all X, y, w, ids"""
        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_seg = np.squeeze(np.vstack(all_seg))
        all_y_one_hot = np.squeeze(np.vstack(all_y_one_hot))
        assert all_X.shape[0] == all_y.shape[0] == all_seg.shape[0] == all_y_one_hot.shape[0]

        train_dataset = dict()
        train_dataset['X'] = all_X
        train_dataset['y'] = all_y
        train_dataset['seg_mask'] = all_seg
        train_dataset['one_hot_y'] = all_y_one_hot

        metadata_rows = []
        all_X, all_y, all_seg, all_y_one_hot = [], [], [], []
        for shard_num, (shard, y, seg, y_1h) in enumerate(self.get_shards(TEST_FILES, shard_size)):
            """shard is ndarray of 3D mesh point 3-dimension, [pc_id, point_id, (x,y,z)]
                y is the class id for each sample (3D mesh), int type [pc_id, part_id of point]
            """

            print('Featurizing Testing Data , Shard - %d' % shard_num)
            X, seg = self.featurize_shard(shard, seg)

            """ save shard (x,y,ids, w)"""
            basename = "shard-%d" % shard_num
            metadata_rows.append(self.write_data_to_disk(test_dir, basename, X, y, seg, y_1h))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_seg.append(seg)
            all_y_one_hot.append(y_1h)

        """ save the meta data to local """
        meta_data2 = list()
        meta_data2.append(metadata_rows)
        with open(os.path.join(test_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta_data2, f)

        """ return a Dataset contains all X, y, w, ids"""
        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_seg = np.squeeze(np.vstack(all_seg))
        all_y_one_hot = np.squeeze(np.vstack(all_y_one_hot))
        assert all_X.shape[0] == all_y.shape[0] == all_seg.shape[0] == all_y_one_hot.shape[0]

        test_dataset = dict()
        test_dataset['X'] = all_X
        test_dataset['y'] = all_y
        test_dataset['seg_mask'] = all_seg
        test_dataset['one_hot_y'] = all_y_one_hot

        return train_dataset, test_dataset

    def get_shards(self, file_list, shard_size):

        """ shuffle the data files"""
        file_idxs = np.arange(0, len(file_list))
        np.random.shuffle(file_idxs)  # randomly extract data from files

        """read all data"""
        all_data, all_label, all_seg, all_one_hot_label = [], [], [], []
        for fn in file_idxs:
            # load the point cloud from the file, current_data -> data from one file
            current_data, current_label, current_seg = pd.loadDataFile_with_seg(os.path.join(BASE_DIR, file_list[fn]))
            # remove redundant dimensions
            current_label = np.squeeze(current_label)
            current_seg = np.squeeze(current_seg)

            # shuffle data order
            cur_data, cur_labels, order = pd.shuffle_data(current_data, np.squeeze(current_label))
            current_seg = current_seg[order, ...]

            # create one-hot class label
            cur_labels_one_hot = self.convert_label_to_one_hot(current_label)
            cur_labels_one_hot = np.squeeze(cur_labels_one_hot)

            all_data.append(current_data)
            all_label.append(current_label)
            all_seg.append(current_seg)
            all_one_hot_label.append(cur_labels_one_hot)

        """ create numpy for all data and label"""
        all_data = np.squeeze(np.vstack(all_data))
        all_seg = np.squeeze(np.vstack(all_seg))
        all_one_hot_label = np.squeeze(np.vstack(all_one_hot_label))
        all_label = np.squeeze(np.hstack(all_label))
        assert all_label.shape[0] == all_data.shape[0] == all_seg.shape[0] == all_one_hot_label.shape[0]

        "generate shard data, label, seg_mask, one_hot labels"
        shard_num = 0

        start_idx = shard_num * shard_size
        end_idx = (shard_num + 1) * shard_size
        while end_idx < all_data.shape[0]:
            sel_data = all_data[start_idx:end_idx, :, :]  # ready data
            sel_labels = all_label[start_idx:end_idx]       # corresponding labels
            sel_seg = all_seg[start_idx:end_idx, :]
            sel_one_hot_labels = all_one_hot_label[start_idx:end_idx, :]
            shard_num += 1
            start_idx = shard_num * shard_size
            end_idx = (shard_num + 1) * shard_size
            yield sel_data, sel_labels, sel_seg, sel_one_hot_labels
        # end exceed the bound of pc_names, return all
        yield all_data[start_idx:, :, :], all_label[start_idx:], all_seg[start_idx:, :], all_one_hot_label[start_idx:, :]

    @staticmethod
    def write_data_to_disk(
                           data_dir,
                           basename,
                           X=None,
                           y=None,
                           seg=None,
                           y_1h=None):

        out_X = "%s-X.joblib" % basename
        save_to_disk(X, os.path.join(data_dir, out_X))

        out_y = "%s-y.joblib" % basename
        save_to_disk(y, os.path.join(data_dir, out_y))

        out_seg = "%s-seg.joblib" % basename
        save_to_disk(seg, os.path.join(data_dir, out_seg))

        out_y_1h = "%s-y1h.joblib" % basename
        save_to_disk(y_1h, os.path.join(data_dir, out_y_1h))

        # note that this corresponds to the _construct_metadata column order
        return {'basename': basename, 'X': out_X, 'y': out_y, 'seg_mask': out_seg, 'one_hot_y': out_y_1h}

    def convert_label_to_one_hot(self, labels):
        label_one_hot = np.zeros((labels.shape[0], self.n_classes))
        for idx in range(labels.shape[0]):
            label_one_hot[idx, labels[idx]] = 1
        return label_one_hot

    def featurize_shard(self, shard, seg_mask):

        """
        convert ndarray (n-sample, n_point of sample, 3-d)
        :param shard: ndarray, point cloud raw data
        :return: graph object for each sample
        """
        cluster_num = CLUSTER_NUM  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []  # to save the graph object
        S = []  # list of new mask
        n_samples = shard.shape[0]
        for pc_id in range(n_samples):
            # iterate each pc in shard
            P = np.squeeze(shard[pc_id, :, :])
            mask = np.squeeze(seg_mask[pc_id, :])
            if P.shape[0] > cluster_num:
                # clustering on original cloud
                cluster_indicators = clusterer.fit_predict(P)  # labels of original points
                new_points = []
                Point_mask = []
                for i in range(cluster_num):
                    """ each cluster center is the new point after clustering"""
                    idx = np.where(cluster_indicators == i)
                    coord = np.mean(P[idx], axis=0)
                    part_id = np.argmax(np.bincount(mask[idx])).astype(np.int32)
                    new_points.append(coord)
                    Point_mask.append(part_id)  # this point at coord belongs to part_id
                new_PC = np.asarray(new_points).astype(np.float32)
                new_mask = np.asarray(Point_mask).astype(np.int32)
                assert new_PC.shape[0] <= cluster_num and new_PC.shape[0] == new_mask.shape[0]
            else:
                # do not need a clustering, use original points
                new_PC = P
                new_mask = mask

            # create graph on point cloud
            adj_list, adj_matrix = self.get_adjacency(new_PC)
            X.append(Graph(new_PC, adj_list, max_deg=cluster_num, min_deg=0))
            S.append(new_mask)

        return np.asarray(X), np.asarray(S),

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

    def load_back_from_disk(self, data_dir, istrain=True):
        """
        load data backas Train/test from disk
        :return: Train/Test STDiskDataset
        """
        """load back metadata_df"""
        meta_data = pickle.load(open(os.path.join(data_dir, 'meta.pkl'), 'rb'))
        metadata_rows = meta_data[0]

        """itershard by loading from disk"""
        all_X, all_y, all_seg, all_y_1h = [], [], [], []

        for _, row in enumerate(metadata_rows):
            # each row is a dict, contains file name for X, y, seg, y_one_hot
            X = np.array(load_from_disk(os.path.join(data_dir, row['X'])))
            y = np.array(load_from_disk(os.path.join(data_dir, row['y'])))
            seg = np.array(load_from_disk(os.path.join(data_dir, row['seg_mask'])))
            y_1h = np.array(load_from_disk(os.path.join(data_dir, row['one_hot_y'])))

            """ stack to list"""
            all_X.append(X)
            all_y.append(y)
            all_seg.append(seg)
            all_y_1h.append(y_1h)

        """ return a Dataset contains all X, y, w, ids"""
        if istrain:
            cut = TRAIN_NUM
        else:
            cut = TEST_NUM
        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_seg = np.squeeze(np.vstack(all_seg))
        all_y_1h = np.squeeze(np.vstack(all_y_1h))
        assert all_X.shape[0] == all_y.shape[0] == all_seg.shape[0] == all_y_1h.shape[0]

        dataset = dict()
        dataset['X'] = all_X[cut:]
        dataset['y'] = all_y[cut:]
        dataset['seg_mask'] = all_seg[cut:]
        dataset['one_hot_y'] = all_y_1h[cut:]

        return dataset

    def load(self):
        """Load chemical datasets. Raw data is given as SMILES format."""

        meta_data = []
        if len(os.listdir(self.processed_data_dir)) != 0:

            print("Loading Saved Data from Disk.......")

            """ pre-defined location for saving the train and test data"""
            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            train = self.load_back_from_disk(data_dir=train_dir, istrain=True)
            test = self.load_back_from_disk(data_dir=test_dir, istrain=False)

            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            max_atom = meta_data[0]

        else:
            print("Loading and Featurizing Data.......")
            # loader = dc.data.CSVLoader(
            #     tasks=self.tasks, smiles_field=self.smiles_field, featurizer=self.Featurizer)
            train, test = self.featurize(shard_size=256)

            """max_atom is the max atom of molecule in all_dataset """
            max_atom = CLUSTER_NUM  # because we do clustering, so it is the cluster center number
            meta_data.append(max_atom)
            with open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta_data, f)

        return train, test, max_atom


