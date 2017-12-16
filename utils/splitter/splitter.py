
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tempfile
from AGCN.utils.dataset import NumpyDataset
from AGCN.utils.save import log


class Splitter(object):
    """
    Abstract base class for chemically aware splits..
    """

    def __init__(self, verbose=False):
        """Creates splitter object."""
        self.verbose = verbose

    def k_fold_split(self, dataset, k, directories=None):
        """Does K-fold split of dataset."""
        log("Computing K-fold split", self.verbose)
        if directories is None:
            directories = [tempfile.mkdtemp() for _ in range(k)]
        else:
            assert len(directories) == k
        fold_datasets = []
        # rem_dataset is remaining portion of dataset
        rem_dataset = dataset

        for fold in range(k):
            # Note starts as 1/k since fold starts at 0. Ends at 1 since fold goes up
            # to k-1.
            frac_fold = 1. / (k - fold)
            fold_dir = directories[fold]
            fold_inds, rem_inds, _ = self.split(
                rem_dataset,
                frac_train=frac_fold,
                frac_valid=1 - frac_fold,
                frac_test=0)
            fold_dataset = rem_dataset.select(fold_inds, fold_dir)
            rem_dir = tempfile.mkdtemp()
            rem_dataset = rem_dataset.select(rem_inds, rem_dir)
            fold_datasets.append(fold_dataset)
        return fold_datasets

    def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True):
        """
        Splits self into train/validation/test sets.

        Returns Dataset objects.
        """
        if isinstance(dataset, NumpyDataset):
            raise ValueError(
              "Only possible with DiskDataset.  NumpyDataset doesn't support .select"
            )
        log("Computing train/valid/test indices", self.verbose)
        train_inds, valid_inds, test_inds = self.split(
            dataset,
            frac_train=frac_train,
            frac_test=frac_test,
            frac_valid=frac_valid,
            log_every_n=log_every_n)
        if train_dir is None:
            train_dir = tempfile.mkdtemp()
        if valid_dir is None:
            valid_dir = tempfile.mkdtemp()
        if test_dir is None:
            test_dir = tempfile.mkdtemp()

        train_dataset = dataset.select(train_inds, train_dir)
        if frac_valid != 0:
            valid_dataset = dataset.select(valid_inds, valid_dir)
        else:
            valid_dataset = None
        test_dataset = dataset.select(test_inds, test_dir)

        return train_dataset, valid_dataset, test_dataset

    def train_test_split(self,
                       dataset,
                       train_dir=None,
                       test_dir=None,
                       frac_train=.6,
                       verbose=True):
        """
        Splits self into train/test sets.
        Returns Dataset objects.
        """
        valid_dir = tempfile.mkdtemp()
        train_dataset, _, test_dataset = self.train_valid_test_split(
            dataset,
            train_dir,
            valid_dir,
            test_dir,
            frac_train=frac_train,
            frac_test=1 - frac_train,
            frac_valid=0.,
            verbose=verbose)
        return train_dataset, test_dataset

    def split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            log_every_n=None,
            verbose=False):
        """
        Stub to be filled in by child classes.
        """
        raise NotImplementedError
