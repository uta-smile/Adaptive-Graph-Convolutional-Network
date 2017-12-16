
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pickle

from AGCN.utils.feature import CircularFingerprint, ConvMolFeaturizer
from AGCN.utils.transformer import BalancingTransformer, NormalizationTransformer
from AGCN.utils.dataset import DiskDataset
from AGCN.utils.splitter import IndexSplitter, ScaffoldSplitter, RandomSplitter
from AGCN.utils.save import load_from_disk


class DataLoader(object):
    """Abstract base class for Data Loader."""
    def __init__(self,
                 dataset_class,
                 dataset_name,
                 file_name=None,
                 tasks=None,
                 n_classes=2,
                 smiles_field='smiles',
                 id_field=None,
                 feature=None,
                 splitter='index',
                 split_frac=[0.8, 0.2],
                 transformer='normalization_w',
                 download_url=None,
                 verbose=True):

        self.package_dir = os.path.join(os.environ["HOME"], 'AGCN/AGCN')
        self.dataset_class = dataset_class  # chemistry, point cloud
        self.dataset_name = dataset_name    # e.g. tox21, clintox
        self.data_dir = os.path.join(os.environ["HOME"], "AGCN/data", self.dataset_class, self.dataset_name)
        self.file_name = file_name  # raw data file name, e.g clintox.csv.gz
        if self.file_name:
            # the data is in a file, create file_dir to use
            self.file_dir = os.path.join(self.data_dir, self.file_name)     # raw data saved directory
        else:
            self.file_dir = None    # file directory not given, Or data is saved as a folder
        self.download_url = download_url
        self.processed_data_dir = os.path.join(self.data_dir, 'processed_data', self.dataset_name)
        self.verbose = verbose

        "confirm data exist"
        # if not os.path.exists(self.file_dir) and self.download_url is not None:
        #     os.system(
        #         'wget -P ' + self.data_dir + self.download_url
        #     )

        "confirm processed data storage directory"
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

        "define task/label names"
        if tasks is None and len(os.listdir(self.processed_data_dir)) == 0:
            # no given tasks and no processed data, need to retrieve from dataset
            dataset = load_from_disk(self.file_dir)
            self.tasks = dataset.columns.values[1:].tolist()
        elif tasks is not None and isinstance(tasks, list):
            # not pre-calculated, but directly given by input
            self.tasks = tasks  # define the label field names
        elif len(os.listdir(self.processed_data_dir)) != 0:
            # already be retrieved and saved at meta data
            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            self.tasks = meta_data[1:]
        else:
            raise ValueError("if task name is given, it must be a list.")

        "define smile field/ ID field"
        self.smiles_field = smiles_field
        # if not specified, smiles is id
        if id_field is None:
            self.id_field = smiles_field
        else:
            self.id_field = id_field

        "define number of class for each task, if not specified, it is 2, binary classification for each task"
        self.n_classes = n_classes

        "define feature extractor. if feature = None, it is image, pointcloud, no need for featurizer"
        self.feature = feature  # feature type -> featurizer function
        if self.feature == 'ECFP':
            self.Featurizer = CircularFingerprint(size=1024)
        elif self.feature == 'MolGraph':
            self.Featurizer = ConvMolFeaturizer()
        else:
            self.Featurizer = None

        "define splitter"
        self.splitter = splitter
        self.split_frac = split_frac

        "define the transformer"
        self.transformer_types = transformer
        self.transformers = []

    def featurize(self, shard_size=1024):
        """
            Featurize provided data files and write to specified location.
        """
        input_files = self.file_dir
        data_dir = self.processed_data_dir
        assert os.path.exists(input_files)
        assert os.path.exists(data_dir)

        if not isinstance(input_files, list):
            input_files = [input_files]

        def shard_generator():
            for shard_num, shard in enumerate(self.get_shards(input_files, shard_size)):

                X, valid_inds = self.featurize_shard(shard)
                ids = shard[self.id_field].values
                ids = ids[valid_inds]
                if len(self.tasks) > 0:
                    # find task labels and weights iff they exist.
                    y, w = DataLoader.convert_df_to_numpy(shard, self.tasks)
                    # Filter out examples where has no valid labels.
                    y, w = (y[valid_inds], w[valid_inds])
                    assert len(X) == len(ids) == len(y) == len(w)
                    X, y, w, ids = DataLoader.remove_2small(X, y, w, ids)
                else:
                    # For prospective data where results are unknown, it makes
                    # no sense to have y values or weights.
                    y, w = (None, None)
                    assert len(X) == len(ids)
                yield X, y, w, ids

        return DiskDataset.create_dataset(
            shard_generator(), data_dir, self.tasks, verbose=self.verbose)

    @staticmethod
    def remove_2small(X, y, w, ids):
        """ remove the graph that has nodes < 3"""
        sel_list = []
        for i, mol in enumerate(list(X)):
            """remove those has no valid Laplacian"""
            if mol.has_Lap:
                sel_list += [i]
        X = X[sel_list]
        y = y[sel_list]
        w = w[sel_list]
        ids = ids[sel_list]
        return X, y, w, ids

    @staticmethod
    def convert_df_to_numpy(df, tasks):
        """Transforms a dataframe containing deepchem input into numpy arrays"""
        n_samples = df.shape[0]
        n_tasks = len(tasks)

        y = np.hstack(
            [np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])

        w = np.ones((n_samples, n_tasks))
        missing = np.zeros_like(y).astype(int)

        for ind in range(n_samples):
            for task in range(n_tasks):
                if y[ind, task] == "":
                    missing[ind, task] = 1

        # Set missing data to have weight zero
        for ind in range(n_samples):
            for task in range(n_tasks):
                if missing[ind, task]:
                    y[ind, task] = 0.
                    w[ind, task] = 0.

        return y.astype(float), w.astype(float)

    @staticmethod
    def find_max_atom(all_dataset):
        # find the maximum atom number in whole datasests
        max_n = 0
        if not isinstance(all_dataset, list):
            all_dataset = [all_dataset]
        for subset in list(all_dataset):
            data = subset.X
            for mol in data:
                max_n = max(mol.n_node, max_n)
        return max_n

    def get_shards(self, input_files, shard_size):
        """Stub for children classes."""
        raise NotImplementedError

    def featurize_shard(self, shard):
        """Featurizes a shard of an input dataframe."""
        raise NotImplementedError

    def load(self):
        """Load chemical datasets. Raw data is given as SMILES format"""

        meta_data = []
        if len(os.listdir(self.processed_data_dir)) != 0:

            print("Loading Saved Data from Disk.......")

            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            dataset = DiskDataset(data_dir=self.processed_data_dir)
            train = DiskDataset(data_dir=train_dir)
            test = DiskDataset(data_dir=test_dir)

            meta_data = pickle.load(open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'rb'))
            max_atom = meta_data[0]

            print("Transforming Data.")
            if not self.transformer_types:
                if self.transformer_types == 'normalization_y':
                    self.transformers += [
                        NormalizationTransformer(transform_y=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'normalization_w':
                    self.transformers += [
                        NormalizationTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_w':
                    self.transformers += [
                        BalancingTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_y':
                    self.transformers += [
                        BalancingTransformer(transform_y=True, dataset=dataset)
                    ]
                else:
                    ValueError("Transformer type Not defined!{}".format(self.transformer_types))

        else:
            print("Loading and Featurizing Data.......")
            # loader = dc.data.CSVLoader(
            #     tasks=self.tasks, smiles_field=self.smiles_field, featurizer=self.Featurizer)
            dataset = self.featurize(shard_size=2048)

            print("Transforming Data.")
            if not self.transformer_types:
                if self.transformer_types == 'normalization_y':
                    self.transformers += [
                        NormalizationTransformer(transform_y=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'normalization_w':
                    self.transformers += [
                        NormalizationTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_w':
                    self.transformers += [
                        BalancingTransformer(transform_w=True, dataset=dataset)
                    ]
                elif self.transformer_types == 'balancing_y':
                    self.transformers += [
                        BalancingTransformer(transform_y=True, dataset=dataset)
                    ]
                else:
                    ValueError("Transformer type Not defined!{}".format(self.transformer_types))

            if len(self.transformers) > 0:
                for transformer in self.transformers:
                    # pass dataset through maybe more than one transformer
                    dataset = transformer.transform(dataset)

            """max_atom is the max atom of molecule in all_dataset """
            max_atom = self.find_max_atom(dataset)
            meta_data.append(max_atom)
            meta_data.extend(self.tasks)
            with open(os.path.join(self.processed_data_dir, 'meta.pkl'), 'wb') as f:
                pickle.dump(meta_data, f)

            """
            Split Dataset
            """
            print("Splitting Date to Train/Validation/Testing")
            splitters = {
                'index': IndexSplitter(),
                'random': RandomSplitter(),
                'scaffold': ScaffoldSplitter()
            }

            if self.splitter not in splitters:
                raise ValueError("Splitter not defined!")
            else:
                splitter = splitters[self.splitter]

            # create processed dirs as train, valid, test
            train_dir = os.path.join(self.processed_data_dir, 'train')
            test_dir = os.path.join(self.processed_data_dir, 'test')

            print("Saving Data at %s...", self.processed_data_dir)
            train, test = splitter.train_test_split(
                dataset,
                train_dir=train_dir,
                test_dir=test_dir,
                frac_train=self.split_frac[0],
            )

        return self.tasks, (train, test), self.transformers, max_atom
