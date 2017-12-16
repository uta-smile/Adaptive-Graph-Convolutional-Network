
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
# import cPickle
import pickle
import numpy as np
from os import listdir
from PIL import Image
import sklearn.cluster as sc
import gzip

from AGCN.utils.dataset import DiskDataset
from AGCN.utils.data_loader import DataLoader, PointcloudLoader
from AGCN.models import Graph


class ImageCIFARLoader(DataLoader):

    @staticmethod
    def one_hot_label(y):
        # generate one hot labels y as ndarray
        assert isinstance(y, np.ndarray)
        class_num = np.unique(y).shape[0]
        onehot = np.zeros((y.shape[0], class_num))
        onehot[np.arange(y.shape[0]), y.tolist()] = 1
        return onehot.astype(np.float32)

    def featurize_shard(self, shard):
        assert isinstance(shard, np.ndarray)

        cluster_num = 30  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []
        for img in shard.tolist():
            # each row is a RGB image 32*32*3
            img_size = 32*32
            red = img[:img_size]
            green = img[img_size: img_size * 2]
            blue = img[img_size * 2:]
            # coordinate (row, col). left-up (0, 0) to right-bottom (n, n)
            y, x = np.meshgrid(np.arange(32), np.arange(32))
            row, col = x.reshape(32*32, ), y.reshape(32*32, )
            node_features = np.vstack((row, col, red, green, blue)).T  # 32*32 * 5 as features {location, color},

            """This graph, is too large, has 1024 node, whose adjacency matrix is 1020 * 1024"""
            new_nodes = []
            cluster_indicators = clusterer.fit_predict(node_features)
            for i in range(cluster_num):
                idx = np.where(cluster_indicators == i)
                coord = np.mean(node_features[idx], axis=0)
                new_nodes.append(coord)
            new_node_features = np.asarray(new_nodes).astype(np.float32)

            adj_list, adj_matrix = PointcloudLoader.get_adjacency(new_node_features)
            X.append(Graph(new_node_features, adj_list, max_deg=cluster_num, min_deg=0))

        return np.asarray(X), np.asarray([True] * len(X))

    def visualize_shard(self, shard):
        # for testing, display image
        assert isinstance(shard, np.ndarray)
        # shard is a buffer of images
        selected_id = 0
        for id, img in enumerate(shard.tolist()):
            # each row is a RGB image 32*32*3
            if id == selected_id:   # only display this one
                img_size = 32 * 32
                img = np.asarray(img)
                RGB = np.zeros((32, 32, 3))
                RGB[:, :, 0], RGB[:, :, 1], RGB[:, :, 2] = img[:img_size].reshape((32, 32)), \
                                                           img[img_size: img_size * 2].reshape((32, 32)), \
                                                           img[img_size * 2:].reshape((32, 32))
                rgb_img = Image.fromarray(RGB.astype(np.uint8))
                rgb_img.show()

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
            for shard_num, (data_shard, label_shard) in enumerate(
                    self.get_shards(input_folders, shard_size)):

                # self.visualize_shard(data_shard)
                X, valid_inds = self.featurize_shard(data_shard)
                ids = np.arange(shard_num * shard_size, shard_num * shard_size + len(X))   # for CIFAR, ids is indexing
                # Featurize task results iff they exist.
                y = self.one_hot_label(label_shard.astype(np.int32))
                w = np.ones(y.shape)
                # Filter out examples where featurization failed.
                y, w = (y[valid_inds], w[valid_inds])
                assert len(X) == len(ids) == len(y) == len(w)
                print ("finished shard No. %d", shard_num)
                yield X, y, w, ids

        return DiskDataset.create_dataset(
            shard_generator(), data_dir, self.tasks, verbose=self.verbose)

    def get_shards(self, input_folders, shard_size=2048):
        """ read one shard of data from input folders, which may contains more than one file"""
        shard_num = 1
        for folder in input_folders:    # iterate folders, but usually one folder
            files_list = [f for f in listdir(folder) if f[:4] == 'data']

            data_batch_l = []   # store batches
            label_batch_l = []
            for fid, f in enumerate(files_list):
                f_dir = os.path.join(folder, f)
                with open(f_dir, 'rb') as fo:
                    batch = pickle.load(fo)
                    data_batch_l.append(batch['data'])
                    label_batch_l.append(batch['labels'])
            # stack into one single array
            data = np.vstack(tuple(data_batch_l))
            label = np.hstack(tuple(label_batch_l))
            assert label.shape[0] == data.shape[0]

            if shard_size is None:  # give all in once
                yield data.astype(np.float32), label.astype(np.float32)
            else:
                end = int(shard_num * shard_size)
                start = int((shard_num - 1) * shard_size)
                while end <= data.shape[0]:
                    data_shard = data[start: end]
                    label_shard = label[start: end]
                    shard_num += 1
                    end = int(shard_num * shard_size)
                    start = int((shard_num - 1) * shard_size)
                    yield data_shard.astype(np.float32), label_shard.astype(np.float32)
                # end exceed the bound of data row dimensions, return all rest
                yield data[start:].astype(np.float32), label[start:].astype(np.float32)


class ImageFashionLoader(ImageCIFARLoader):
    """
    Note: can be also used for MNIST dataset
    """
    def get_shards(self, input_folders, shard_size=2048):
        """
        In the input folders, they have:
        1. train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz, 60000 samples/labels
        2. t10k-images-idx3-ubyte.gz and t10k-labels-idx3-ubyte.gz, 1000 samples/labels
        We mix them two and re-split as train, validation and tes
        :param input_folders:
        :param shard_size:
        :return: one shard of data and labels
        """
        shard_num = 1
        for folder in input_folders:  # iterate folders, but usually one folder
            data_file_list = [f for f in listdir(folder) if 'images' in f.split('-')]   # if file name has 'images'
            label_file_list = [f for f in listdir(folder) if 'labels' in f.split('-')]

            data_batch_l = []  # store batches
            label_batch_l = []
            for fid, f in enumerate(label_file_list):
                with gzip.open(os.path.join(folder, f), 'rb') as fo:
                    label = np.frombuffer(fo.read(), dtype=np.uint8, offset=8)
                    label_batch_l.append(label)

            for fid, f in enumerate(data_file_list):
                with gzip.open(os.path.join(folder, f), 'rb') as fo:
                    n_img = 60000 if 'train' in f.split('-') else 10000
                    data = np.frombuffer(fo.read(), dtype=np.uint8, offset=16).reshape(n_img, 28*28)
                    data_batch_l.append(data)

            # stack into one single array
            data = np.vstack(tuple(data_batch_l))
            label = np.hstack(tuple(label_batch_l))
            assert label.shape[0] == data.shape[0]

            # return shard by shard
            if shard_size is None:  # if shard size is not defined, give all in once
                yield data.astype(np.float32), label.astype(np.float32)
            else:
                end = int(shard_num * shard_size)
                start = int((shard_num - 1) * shard_size)
                while end <= data.shape[0]:
                    data_shard = data[start: end]
                    label_shard = label[start: end]
                    shard_num += 1
                    end = int(shard_num * shard_size)
                    start = int((shard_num - 1) * shard_size)
                    yield data_shard.astype(np.float32), label_shard.astype(np.float32)
                # end exceed the bound of data row dimensions, return all rest
                yield data[start:].astype(np.float32), label[start:].astype(np.float32)

    def featurize_shard(self, shard):
        assert isinstance(shard, np.ndarray)

        cluster_num = 30  # min_node in this folder is 13 precomputed
        clusterer = sc.AgglomerativeClustering(n_clusters=cluster_num)

        X = []
        for img in shard.tolist():
            # each row is a RGB image 32*32*3
            img_size = 28*28
            greyscale = img[:img_size]
            # coordinate (row, col). left-up (0, 0) to right-bottom (n, n)
            y, x = np.meshgrid(np.arange(28), np.arange(28))
            row, col = x.reshape(28*28, ), y.reshape(28*28, )
            node_features = np.vstack((row, col, greyscale)).T

            """This graph, is too large, has 1024 node, whose adjacency matrix is 1020 * 1024"""
            new_nodes = []
            cluster_indicators = clusterer.fit_predict(node_features)
            for i in range(cluster_num):
                idx = np.where(cluster_indicators == i)
                coord = np.mean(node_features[idx], axis=0)
                new_nodes.append(coord)
            new_node_features = np.asarray(new_nodes).astype(np.float32)

            adj_list, adj_matrix = PointcloudLoader.get_adjacency(new_node_features)
            X.append(Graph(new_node_features, adj_list, max_deg=cluster_num, min_deg=0))

        return np.asarray(X), np.asarray([True] * len(X))
