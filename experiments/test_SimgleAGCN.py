"""
create a demo experiment using SimpleAGCN network on Tox21 data.

1. data - Tox21
2. network - SimpleAGCN.
"""

import numpy as np

from AGCN.models.networks import SimpleAGCN
from AGCN.utils.data_loader import SMILESLoader
from AGCN.experiments.dataloader_configs import loaderconfig_dict
from AGCN.experiments.hyper_parameters import exp_hps
from AGCN.models.tf_modules.metrics import Metric, roc_auc_score, precision_score, recall_score


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

# load hyper-parameters
hyper_parameters = exp_hps['test_SimpleAGCN']
print("Hyper-parameter Loaded!  \n")


# create network
network = SimpleAGCN(train_data,
                     valid_data,
                     test_data,
                     max_atom,
                     tasks,
                     hyper_parameters,
                     transformers,
                     metrics)

network.train()
network.final_evaluate()
network.plot_loss_curve()
network.plot_score_curves('task_averaged-precision_score')





