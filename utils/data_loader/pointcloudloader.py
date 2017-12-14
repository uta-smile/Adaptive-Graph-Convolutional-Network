
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
from AGCN.models import Graph
from AGCN.utils.save import save_to_disk, load_from_disk


NUM_CLUSTER = 50


class PointcloudLoader(DataLoader):
    """
    This class of loader is for Point Cloud data, which is usually lots of binary files stay in
    a folder. Each binary file is a point cloud object, reading it, you get a nddarray of 3D points, each rwo of which
    is a point in laser receivers.
    Input: folder directory
    Output: ConvMol instance for each point cloud object. Features is intensity, adjacency matrix, degree
    are manually learned and set by clustering.
    """

    def featurize(self, shard_size=2048):
        """
        Args:
            input_folder: folder name of binary files (graph object)
            data_dir: where to save generated data frame
            shard_size: number of graphs in one shard
        Returns:
        """
        input_folder = self.file_dir    # where to put the folders of PC objects
        data_dir = self.processed_data_dir  # where to store the processed sharded data
        assert os.path.exists(input_folder)
        assert os.path.exists(data_dir)

        if not isinstance(input_folder, list):
            input_folders = [input_folder]
        else:
            input_folders = input_folder

        metadata_rows = []
        all_X, all_y, all_w, all_ids = [], [], [], []
        for shard_num, (shard, pc_dir) in enumerate(self.get_shards(input_folders, shard_size)):

            X, valid_inds = self.featurize_shard(shard, pc_dir)
            ids = np.asarray(shard)  # ids is the
            # Featurize task results iff they exist.
            # y is 1-d label, just tell the class id, 0-25 for sydney
            _, y = self.get_labels(shard, pc_dir)
            w = np.ones((y.shape[0], ))
            # Filter out examples where featurization failed.
            y, w = (y[valid_inds], w[valid_inds])
            assert len(X) == len(ids) == len(y) == len(w)

            """ save shard (x,y,ids, w)"""
            basename = "shard-%d" % shard_num
            metadata_rows.append(self.write_data_to_disk(self.processed_data_dir, basename, X, y, w, ids))

            all_X.append(X)
            all_y.append(y)
            all_w.append(w)
            all_ids.append(ids)

        """ save the meta data of training dataset to local """
        meta_data1 = list()
        meta_data1.append(metadata_rows)
        with open(os.path.join(self.processed_data_dir, 'meta_files.pkl'), 'wb') as f:
            pickle.dump(meta_data1, f)

        all_X = np.concatenate(all_X)
        all_y = np.squeeze(np.concatenate(all_y))
        all_w = np.squeeze(np.concatenate(all_w))
        all_ids = np.squeeze(np.concatenate(all_ids))

        # shuffle data
        order = np.arange(0, all_X.shape[0])
        np.random.shuffle(order)
        all_X = all_X[order]
        all_y = all_y[order]
        all_w = all_w[order]
        all_ids = all_ids[order]

        # split them as train and test
        n_train = int(self.split_frac[0] * all_X.shape[0])

        train_dataset = dict()
        train_dataset['X'] = all_X[:n_train]
        train_dataset['y'] = all_y[:n_train]
        train_dataset['w'] = all_w[:n_train]
        train_dataset['ids'] = all_ids[:n_train]

        test_dataset = dict()
        test_dataset['X'] = all_X[n_train:]
        test_dataset['y'] = all_y[n_train:]
        test_dataset['w'] = all_w[n_train:]
        test_dataset['ids'] = all_ids[n_train:]

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

    def get_shards(self, input_folders, shard_size):
        shard_num = 1
        for pc_dir in input_folders:
            # pc_dir is the directory to point cloud files
            if shard_size is None:
                pc_names = [f for f in listdir(pc_dir) if isfile(join(pc_dir, f)) and f[-3:] == 'bin']
                yield pc_names, pc_dir
            else:
                pc_names = [f for f in listdir(pc_dir) if isfile(join(pc_dir, f)) and f[-3:] == 'bin']
                end = int(shard_num * shard_size)
                start = int((shard_num - 1) * shard_size)
                while end <= len(pc_names):
                    pc_sequences = pc_names[start: end]
                    shard_num += 1
                    end = int(shard_num * shard_size)
                    start = int((shard_num - 1) * shard_size)
                    yield pc_sequences, pc_dir
                # end exceed the bound of pc_names, return all
                yield pc_names[start:], pc_dir

    def get_labels(self, shard, pc_dir):
        """
        set one -hot labels for each samples,
        which equals to set a multi-tasks prediction, each class is a task.
        weight for each tasks each sample is always 1

        Args:
            shard: list of names of binary file which contains the class name
            pc_dir: directory where sample saved

        Returns: one-hot labels, weight vectors on each task
        """
        classes_byitems = []
        # count how many classes in dataset
        for f in [f for f in listdir(pc_dir) if isfile(join(pc_dir, f)) and f[-3:] == 'bin']:
            classes_byitems.append(f.split('.')[0])  # first word is class name
        classes_num = len(set(classes_byitems))
        # one class , e.g. car, to one class id
        label2classes = dict(zip(list(set(classes_byitems)), np.arange(classes_num)))

        y = []
        for pc in shard:
            # parse the sample label
            class_name = pc.split('.')[0]
            class_id = label2classes[class_name]
            y.append(class_id)

        one_hot_labels = np.zeros((len(shard), len(label2classes)))     # prepare a matrix
        one_hot_labels[np.arange(len(shard)), y] = 1                    # set the on-hot
        return one_hot_labels, np.asarray(y).astype(np.int32)

    def featurize_shard(self, shard, pc_dir=None, down_sample=False):
        # create graph object for each point cloud object

        """
            binary file format
        """
        names = ['t', 'intensity', 'id',
                 'x', 'y', 'z',
                 'azimuth', 'range', 'pid']

        formats = ['int64', 'uint8', 'uint8',
                   'float32', 'float32', 'float32',
                   'float32', 'float32', 'int32']

        binType = np.dtype(dict(names=names, formats=formats))

        cluster_num = NUM_CLUSTER  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []
        for pc_file in shard:
            # iterate each pc in shard
            data = np.fromfile(os.path.join(pc_dir, pc_file), binType)
            # read original points and intensity
            P = np.vstack([data['x'], data['y'], data['z']]).T
            intensity = np.asarray(data['intensity']).T
            if down_sample:
                if P.shape[0] > cluster_num:
                    # clustering on original cloud
                    cluster_indicators = clusterer.fit_predict(P)  # labels of original points
                    new_points, new_intensity = [], []
                    for i in range(cluster_num):
                        idx = np.where(cluster_indicators == i)
                        coord = np.mean(P[idx], axis=0)
                        intent = np.mean(intensity[idx], axis=0)
                        new_points.append(coord)
                        new_intensity.append(intent)
                    new_PC = np.asarray(new_points).astype(np.float32)
                    new_intentities = np.asarray(new_intensity).astype(np.float32)
                else:
                    # do not do clustering, use original points and intensity
                    new_PC = P
                    new_intentities = intensity

                # create graph on point cloud
                node_features = np.hstack([new_PC, np.expand_dims(new_intentities, 1)])
                adj_list, adj_matrix = self.get_adjacency(node_features)
                X.append(Graph(node_features, adj_list, max_deg=cluster_num, min_deg=0))
            else:
                """use original samples"""
                node_features = np.hstack([P, np.expand_dims(intensity, 1)])
                adj_list, adj_matrix = self.get_adjacency(node_features)
                X.append(Graph(node_features, adj_list, max_deg=node_features.shape[0], min_deg=0))

        return np.asarray(X), np.asarray([True] * len(X))

    @staticmethod
    def get_adjacency(points):
        n_p = points.shape[0]

        # pre-learn the distance matrix, use mean as threshold to generate adjacency matrix, i.e. who connect who
        all_dist = []
        for i in range(n_p):
            for j in range(i + 1):
                all_dist += [np.linalg.norm(points[i] - points[j])]
                # input points are new points after clustering
        sparse_ratio = 0.1  # this ratio is to the percentage of points in matrix that finally has non-zero values
        cut_off_idx = int(n_p * sparse_ratio)
        d_lim = np.sort(all_dist)[-cut_off_idx]

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
        meta_data = pickle.load(open(os.path.join(data_dir, 'meta_files.pkl'), 'rb'))
        metadata_rows = meta_data[0]

        """itershard by loading from disk"""
        all_X, all_y, all_w, all_ids = [], [], [], []

        for _, row in enumerate(metadata_rows):
            # each row is a dict, contains file name for X, y, seg, y_one_hot
            X = np.array(load_from_disk(os.path.join(data_dir, row['X'])))
            y = np.array(load_from_disk(os.path.join(data_dir, row['y'])))
            w = np.array(load_from_disk(os.path.join(data_dir, row['w'])))
            ids = np.array(load_from_disk(os.path.join(data_dir, row['ids'])))

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

        # shuffle data
        order = np.arange(0, all_X.shape[0])
        np.random.shuffle(order)
        all_X = all_X[order]
        all_y = all_y[order]
        all_w = all_w[order]
        all_ids = all_ids[order]

        # split them as train and test
        n_train = int(self.split_frac[0] * all_X.shape[0])

        train_dataset = dict()
        train_dataset['X'] = all_X[:n_train]
        train_dataset['y'] = all_y[:n_train]
        train_dataset['w'] = all_w[:n_train]
        train_dataset['ids'] = all_ids[:n_train]

        test_dataset = dict()
        test_dataset['X'] = all_X[n_train:]
        test_dataset['y'] = all_y[n_train:]
        test_dataset['w'] = all_w[n_train:]
        test_dataset['ids'] = all_ids[n_train:]

        return train_dataset, test_dataset

    def load(self):
        """Load chemical datasets. Raw data is given as SMILES format"""

        meta_data = []
        if len(os.listdir(self.processed_data_dir)) != 0:

            print("Loading Saved Data from Disk.......")

            train, test = self.load_back_from_disk(data_dir=self.processed_data_dir)

            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            max_atom = meta_data[0]

        else:
            print("Loading and Featurizing Data.......")

            train, test = self.featurize(shard_size=2048)

            """max_atom is the max atom of molecule in all_dataset """
            max_atom = NUM_CLUSTER
            meta_data.append(max_atom)
            meta_data.extend(self.tasks)
            with open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta_data, f)

        return self.tasks, (train, test), self.transformers, max_atom


