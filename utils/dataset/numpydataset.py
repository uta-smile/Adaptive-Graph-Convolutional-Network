
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from AGCN.utils.dataset import Dataset


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays."""

    def __init__(self, X, y=None, w=None, ids=None):
        n_samples = len(X)
        # The -1 indicates that y will be reshaped to have length -1
        if n_samples > 0:
            if y is not None:
                y = np.reshape(y, (n_samples, -1))
                if w is not None:
                    w = np.reshape(w, (n_samples, -1))
            else:
                # Set labels to be zero, with zero weights
                y = np.zeros((n_samples, 1))
                w = np.zeros_like(y)
        self.n_tasks = y.shape[1]
        if ids is None:
            ids = np.arange(n_samples)
        if w is None:
            w = np.ones_like(y)
        self._X = X
        self._y = y
        self._w = w
        self._ids = np.array(ids, dtype=object)

    def __len__(self):
        """
        Get the number of elements in the dataset.
        """
        return len(self._y)

    def get_shape(self):
        """Get the shape of the dataset.

        Returns four tuples, giving the shape of the X, y, w, and ids arrays.
        """
        return self._X.shape, self._y.shape, self._w.shape, self._ids.shape

    def get_task_names(self):
        """Get the names of the tasks associated with this dataset. Here 1- n_tasks. No actual names for numpydataset"""
        return np.arange(self._y.shape[1])

    @property
    def X(self):
        """Get the X vector for this dataset as a single numpy array."""
        return self._X

    @property
    def y(self):
        """Get the y vector for this dataset as a single numpy array."""
        return self._y

    @property
    def ids(self):
        """Get the ids vector for this dataset as a single numpy array."""
        return self._ids

    @property
    def w(self):
        """Get the weight vector for this dataset as a single numpy array."""
        return self._w

    def iterbatches(self,
                    batch_size=None,
                    epoch=0,
                    deterministic=False,
                    pad_batches=False):
        """Get an object that iterates over minibatches from the dataset.

        Each minibatch is returned as a tuple of four numpy arrays: (X, y, w, ids).
        """

        def iterate(dataset, batch_size, deterministic, pad_batches):
            n_samples = dataset._X.shape[0]
            if not deterministic:
                sample_perm = np.random.permutation(n_samples)
            else:
                sample_perm = np.arange(n_samples)
            if batch_size is None:
                batch_size = n_samples
            interval_points = np.linspace(
                0, n_samples, np.ceil(float(n_samples) / batch_size) + 1, dtype=int)
            for j in range(len(interval_points) - 1):
                indices = range(interval_points[j], interval_points[j + 1])
                perm_indices = sample_perm[indices]
                X_batch = dataset._X[perm_indices]
                y_batch = dataset._y[perm_indices]
                w_batch = dataset._w[perm_indices]
                ids_batch = dataset._ids[perm_indices]
                if pad_batches:
                    (X_batch, y_batch, w_batch, ids_batch) = Dataset.pad_batch(
                        batch_size, X_batch, y_batch, w_batch, ids_batch)
                yield (X_batch, y_batch, w_batch, ids_batch)

        return iterate(self, batch_size, deterministic, pad_batches)

    def itersamples(self):
        """Get an object that iterates over the samples in the dataset.

        Example:

        >>> dataset = NumpyDataset(np.ones((2,2)))
        >>> for x, y, w, id in dataset.itersamples():
        ...   print(x, y, w, id)
        [ 1.  1.] [ 0.] [ 0.] 0
        [ 1.  1.] [ 0.] [ 0.] 1
        """
        n_samples = self._X.shape[0]
        return ((self._X[i], self._y[i], self._w[i], self._ids[i])
                for i in range(n_samples))

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
        newx, newy, neww = fn(self._X, self._y, self._w)
        return NumpyDataset(newx, newy, neww, self._ids[:])
