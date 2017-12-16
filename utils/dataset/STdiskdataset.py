from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tempfile
import time
import shutil
import pandas as pd
import random

from AGCN.utils.dataset import DiskDataset
from AGCN.utils.save import save_to_disk, load_from_disk, log


class STDiskDataset(DiskDataset):
    """Single task DiskDataset, for PointCloud data, extract from folders, split to train, test"""

    def __init__(self, data_dir, n_classes, verbose=True):
        """
        Turns featurized dataframes into numpy files, writes them & metadata to disk.
        """
        self.data_dir = data_dir
        self.verbose = verbose
        self.n_classes = n_classes

        log("Loading dataset from disk.", self.verbose)
        if os.path.exists(self._get_metadata_filename()):
            (self.tasks,
             self.metadata_df) = load_from_disk(self._get_metadata_filename())
        else:
            raise ValueError("No metadata found on disk.")

    @staticmethod
    def create_dataset(shard_generator, data_dir=None, n_classes=2, tasks=[], verbose=True):
        """Creates a new DiskDataset

        Parameters
        ----------
        shard_generator: Iterable
          An iterable (either a list or generator) that provides tuples of data
          (X, y, L, w, ids). Each tuple will be written to a separate shard on disk.
        data_dir: str
          Filename for data directory. Creates a temp directory if none specified.
        tasks: list
          List of tasks for this dataset.
        n_classes:
        verbose:
        """
        if data_dir is None:
            data_dir = tempfile.mkdtemp()
        elif not os.path.exists(data_dir):
            os.makedirs(data_dir)

        metadata_rows = []
        time1 = time.time()
        for shard_num, (X, y, w, ids) in enumerate(shard_generator):
            basename = "shard-%d" % shard_num

            metadata_rows.append(
                STDiskDataset.write_data_to_disk(data_dir, basename, tasks, X, y, w, ids))

        metadata_df = STDiskDataset._construct_metadata(metadata_rows)
        metadata_filename = os.path.join(data_dir, "metadata.joblib")
        save_to_disk((tasks, metadata_df), metadata_filename)
        time2 = time.time()
        log("TIMING: dataset construction took %0.3f s" % (time2 - time1), verbose)
        return STDiskDataset(data_dir, n_classes, verbose=verbose)

    @staticmethod
    def from_numpy(
            X,
            y,
            w=None,
            ids=None,
            tasks=None,
            data_dir=None,
            n_classes=2,
            verbose=True):

        """Creates a DiskDataset object from specified Numpy arrays."""
        # if data_dir is None:
        #  data_dir = tempfile.mkdtemp()
        n_samples = len(X)
        # The -1 indicates that y will be reshaped to have length -1
        if n_samples > 0:
            y = np.squeeze(y)
            if w is not None:
                w = np.squeeze(w)
        if ids is None:
            ids = np.arange(n_samples)
        if w is None:
            w = np.ones_like(y)
        if tasks is None:
            tasks = np.arange(1)
        # raw_data = (X, y, w, ids)
        return STDiskDataset.create_dataset(
            [(X, y, w, ids)], data_dir=data_dir, n_classes=n_classes, tasks=tasks, verbose=verbose)

    def select(self, indices, select_dir=None):
            """Creates a new dataset from a selection of indices from self. use for splitting dataset to train/test

            Parameters
            ----------
            select_dir: string
              Path to new directory that the selected indices will be copied to.
            indices: list
              List of indices to select.
            """
            if select_dir is not None:
                if not os.path.exists(select_dir):
                    os.makedirs(select_dir)
            else:
                select_dir = tempfile.mkdtemp()
            # Handle edge case with empty indices
            if not len(indices):
                return STDiskDataset.create_dataset(
                    [], data_dir=select_dir, verbose=self.verbose)
            indices = np.array(sorted(indices)).astype(int)
            tasks = self.get_task_names()

            def generator():
                count, indices_count = 0, 0
                for shard_num, (X, y, w, ids) in enumerate(self.itershards()):
                    shard_len = len(X)
                    # Find indices which rest in this shard
                    num_shard_elts = 0
                    while indices[indices_count + num_shard_elts] < count + shard_len:
                        num_shard_elts += 1
                        if indices_count + num_shard_elts >= len(indices):
                            break
                    # Need to offset indices to fit within shard_size
                    shard_inds = indices[indices_count:indices_count + num_shard_elts] - count
                    X_sel = X[shard_inds]
                    y_sel = y[shard_inds]
                    w_sel = w[shard_inds]
                    ids_sel = ids[shard_inds]
                    yield (X_sel, y_sel, w_sel, ids_sel)
                    # Updating counts
                    indices_count += num_shard_elts
                    count += shard_len
                    # Break when all indices have been used up already
                    if indices_count >= len(indices):
                        return

            return STDiskDataset.create_dataset(
                generator(), data_dir=select_dir, n_classes=self.n_classes, tasks=tasks, verbose=self.verbose)

    def pad_batch(self, batch_size, X_b, y_b, w_b, ids_b):
        """Pads batch to have size precisely batch_size elements.
        Fills in batch by wrapping around samples till whole batch is filled.

        # This function is handle single task data
        """
        num_samples = len(X_b)
        if num_samples == batch_size:
            return (X_b, y_b, w_b, ids_b)
        else:
            # By invariant of when this is called, can assume num_samples > 0
            # and num_samples < batch_size
            if len(X_b.shape) > 1:
                feature_shape = X_b.shape[1:]
                X_out = np.zeros((batch_size,) + feature_shape, dtype=X_b.dtype)
            else:
                X_out = np.zeros((batch_size,), dtype=X_b.dtype)

            y_out = np.zeros((batch_size, ), dtype=y_b.dtype)
            w_out = np.zeros((batch_size, ), dtype=w_b.dtype)
            ids_out = np.zeros((batch_size, ), dtype=ids_b.dtype)

            # Fill in batch arrays
            start = 0
            while start < batch_size:
                num_left = batch_size - start
                if num_left < num_samples:
                    increment = num_left
                else:
                    increment = num_samples
                X_out[start:start + increment] = X_b[:increment]
                y_out[start:start + increment] = y_b[:increment]
                w_out[start:start + increment] = w_b[:increment]
                ids_out[start:start + increment] = ids_b[:increment]
                start += increment
            return X_out, y_out, w_out, ids_out

    def iterbatches(self,
                    batch_size=None,
                    epoch=0,
                    deterministic=False,
                    pad_batches=False):

        """Get an object that iterates over minibatches from the dataset.
        Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
        """
        def iterate(dataset):
            num_shards = dataset.get_number_shards()
            if not deterministic:
                shard_perm = np.random.permutation(num_shards)
            else:
                shard_perm = np.arange(num_shards)
            for i in range(num_shards):
                X, y, w, ids = dataset.get_shard(shard_perm[i])
                n_samples = X.shape[0]
                # Handle edge case.
                if n_samples == 0:
                    continue
                if not deterministic:
                    sample_perm = np.random.permutation(n_samples)
                else:
                    sample_perm = np.arange(n_samples)
                if batch_size is None:
                    shard_batch_size = n_samples
                else:
                    shard_batch_size = batch_size
                interval_points = np.linspace(
                    0,
                    n_samples,
                    np.ceil(float(n_samples) / shard_batch_size) + 1,
                    dtype=int)
                for j in range(len(interval_points) - 1):
                    indices = range(interval_points[j], interval_points[j + 1])
                    perm_indices = sample_perm[indices]
                    X_batch = X[perm_indices]

                    if y is not None:
                        y_batch = y[perm_indices]
                    else:
                        y_batch = None

                    if w is not None:
                        w_batch = w[perm_indices]
                    else:
                        w_batch = None

                    ids_batch = ids[perm_indices]
                    if pad_batches:
                        (X_batch, y_batch, w_batch, ids_batch) = self.pad_batch(
                            shard_batch_size, X_batch, y_batch, w_batch, ids_batch)
                    yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self)
