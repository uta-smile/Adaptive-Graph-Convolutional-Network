
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np


class Dataset(object):
    """Abstract base class for datasets defined by X, y, w elements."""

    def __init__(self):
        raise NotImplementedError()

    def __len__(self):
        """
        Get the number of elements in the dataset.
        """
        raise NotImplementedError()

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        raise NotImplementedError()

    def get_task_names(self):
        """Get the names of the tasks associated with this dataset."""
        raise NotImplementedError()

    @property
    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    @property
    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    @property
    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""

        raise NotImplementedError()

    @property
    def w(self):
        """Get the weight vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def iterbatches(self,
                    batch_size=None,
                    epoch=0,
                    deterministic=False,
                    pad_batches=False):
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
        """
        raise NotImplementedError()

    def itersamples(self):
        """Get an object that iterates over the samples in the dataset."""
        raise NotImplementedError()

    def transform(self, fn, **args):
        """Construct a new dataset by applying a transformation to every sample in this dataset.

        The argument is a function that can be called as follows:

        >> newx, newy, neww = fn(x, y, w)

        It might be called only once with the whole dataset, or multiple times with
        different subsets of the data.  Each time it is called, it should transform
        the samples and return the transformed data.

        Parameters
        ----------
        fn: function
          A function to apply to each sample in the dataset

        Returns
        -------
        a newly constructed Dataset object
        """
        raise NotImplementedError()

    def get_statistics(self, X_stats=True, y_stats=True):
        """Compute and return statistics of this dataset."""
        X_means = 0.0
        X_m2 = 0.0
        y_means = 0.0
        y_m2 = 0.0
        n = 0
        for X, y, _, _ in self.itersamples():
            n += 1
            if X_stats:
                dx = X - X_means
                X_means += dx / n
                X_m2 += dx * (X - X_means)
            if y_stats:
                dy = y - y_means
                y_means += dy / n
                y_m2 += dy * (y - y_means)
        if n < 2:
            X_stds = 0.0
            y_stds = 0
        else:
            X_stds = np.sqrt(X_m2 / n)
            y_stds = np.sqrt(y_m2 / n)
        if X_stats and not y_stats:
            return X_means, X_stds
        elif y_stats and not X_stats:
            return y_means, y_stds
        elif X_stats and y_stats:
            return X_means, X_stds, y_means, y_stds
        else:
            return None

    # @staticmethod
    def pad_batch(self, batch_size, X_b, y_b, w_b, ids_b):
        """Pads batch to have size precisely batch_size elements.

        Fills in batch by wrapping around samples till whole batch is filled.
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

            num_tasks = y_b.shape[1]
            y_out = np.zeros((batch_size, num_tasks), dtype=y_b.dtype)
            w_out = np.zeros((batch_size, num_tasks), dtype=w_b.dtype)
            ids_out = np.zeros((batch_size,), dtype=ids_b.dtype)

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

    @staticmethod
    def sparsify_features(X):
        """Extracts a sparse feature representation from dense feature array."""
        n_samples = len(X)
        X_sparse = []
        for i in range(n_samples):
            nonzero_inds = np.nonzero(X[i])[0]
            nonzero_vals = X[i][nonzero_inds]
            X_sparse.append((nonzero_inds, nonzero_vals))
        X_sparse = np.array(X_sparse, dtype=object)
        return X_sparse

    @staticmethod
    def densify_features(X_sparse, num_features):
        """Expands sparse feature representation to dense feature array."""
        n_samples = len(X_sparse)
        X = np.zeros((n_samples, num_features))
        for i in range(n_samples):
            nonzero_inds, nonzero_vals = X_sparse[i]
            X[i][nonzero_inds.astype(int)] = nonzero_vals
        return X
