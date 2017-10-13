
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import sklearn.cluster as sc

from AGCN.utils.data_loader import DataLoader
from AGCN.utils.datatset import DiskDataset
from AGCN.models import Graph


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

        def shard_generator():
            for shard_num, (shard, pc_dir) in enumerate(
                    self.get_shards(input_folders, shard_size)):

                X, valid_inds = self.featurize_shard(shard, pc_dir)
                ids = np.asarray(shard)  # ids is the
                if len(self.tasks) > 0:
                    # Featurize task results iff they exist.
                    y = self.get_labels(shard, pc_dir)
                    w = np.ones(y.shape)
                    # Filter out examples where featurization failed.
                    y, w = (y[valid_inds], w[valid_inds])
                    assert len(X) == len(ids) == len(y) == len(w)

                yield X, y, w, ids

        return DiskDataset.create_dataset(
            shard_generator(), data_dir, self.tasks, verbose=self.verbose)

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

        one_hot_labels = np.zeros((len(shard), len(label2classes)))  # prepare a matrix
        one_hot_labels[np.arange(len(shard)), y] = 1  # set the on-hot
        return one_hot_labels

    def featurize_shard(self, shard, pc_dir=None):
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

        cluster_num = 50  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []
        for pc_file in shard:
            # iterate each pc in shard
            data = np.fromfile(os.path.join(pc_dir, pc_file), binType)
            # read original points and intensity
            P = np.vstack([data['x'], data['y'], data['z']]).T
            intensity = np.asarray(data['intensity']).T

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
            adj_list, adj_matrix = self.get_adjacency(new_PC)
            X.append(Graph(node_features, adj_list, max_deg=cluster_num, min_deg=0))

        return np.asarray(X), np.asarray([True] * len(X))

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

