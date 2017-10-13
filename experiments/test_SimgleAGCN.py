"""
create a demo experiment using SimpleAGCN network on Tox21 data.

1. data - train, valid, test, transformer, max_atom
2. network - defined in models/networks folder.
"""

import numpy as np

from AGCN.models.networks import SimpleAGCN
from AGCN.utils.data_loader import SMILESLoader
from AGCN.experiments.dataloader_configs import loaderconfig_dict
from AGCN.experiments.hyper_parameters import exp_hps
from AGCN.experiments.metrics import Metric, roc_auc_score, precision_score, recall_score


hyper_parameters = exp_hps['test_SimpleAGCN']
print("Hyper-parameter Loaded!  \n")

# create data loader
tox21_loader = loaderconfig_dict['tox21']
loader = SMILESLoader(tox21_loader.dataset_class,
                      tox21_loader.dataset_name,
                      tox21_loader.file_name,
                      tox21_loader.tasks,
                      tox21_loader.smiles_filed,
                      feature='MolGraph',
                      splitter='index',
                      download_url=tox21_loader.download_url)

# load data
tasks, all_dataset, transformers, max_atom = loader.load()
train_data, valid_data, test_data = all_dataset
print("Data Loaded! \n")


# define metrics
metrics = [
    Metric(roc_auc_score, task_averager=np.mean),
    Metric(precision_score, task_averager=np.mean),
    Metric(recall_score, task_averager=np.mean)
]
print("Metrics Loaded! \n")


# create network
network = SimpleAGCN(train_data,
                     valid_data,
                     test_data,
                     max_atom,
                     tasks,
                     hyper_parameters,
                     transformers,
                     metrics)
print("Network Created Successfully! \n")
print("Start Training ...... \n\n")
network.train()





