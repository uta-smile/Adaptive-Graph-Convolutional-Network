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

from AGCN.utils.dataset import DiskDataset, STDiskDataset
from AGCN.utils.save import save_to_disk, load_from_disk, log


class TrainTestDataset(STDiskDataset):
    """
    Still used for Single Task prediction job. Train and test dataset are given separately.
    """
    @staticmethod
    def create_dataset(file_list, data_dir=None, n_classes=2, tasks=[], verbose=True):
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


