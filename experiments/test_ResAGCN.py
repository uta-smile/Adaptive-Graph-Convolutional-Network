"""
create a demo experiment using SimpleAGCN network on Tox21 data.

1. data - Tox21
2. network - SimpleAGCN.
"""

import numpy as np
import os

from AGCN.models.networks import ResAGCN
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
# hyper_parameters = exp_hps['test_ResAGCN']
# print("Hyper-parameter Loaded!  \n")


hyper_parameters_ResAGCN = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 20,
    'n_filters': 64,
    'l_n_filters': [64, 128, 128, 256],
    'final_feature_n': 256,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/AGCN/experiments/results'),
    'model_name': 'ResAGCN',
    'data_name': 'Tox21'
}

# define the search range
lr_list = [0.0005, 0.001, 0.002]


for lr in lr_list:
    # create network
    hyper_parameters_ResAGCN['learning_rate'] = lr
    network = ResAGCN(
                        train_data,
                        valid_data,
                        test_data,
                        max_atom,
                        tasks,
                        hyper_parameters_ResAGCN,
                        transformers,
                        metrics
    )

    network.train()
    network.final_evaluate()
    print('Finish the network training/evaluation with learning rate')
# network.plot_loss_curve()
# network.plot_score_curves('task_averaged-precision_score')

# hyper_parameters_ResAGCN['l_n_filters'] = [64, 64, 64, 64, 64, 64]
#
# for lr in lr_list:
#     # create network
#     hyper_parameters_ResAGCN['learning_rate'] = lr
#     network = ResAGCN(
#                         train_data,
#                         valid_data,
#                         test_data,
#                         max_atom,
#                         tasks,
#                         hyper_parameters_ResAGCN,
#                         transformers,
#                         metrics
#     )
#
#     network.train()
#     network.final_evaluate()
#     print('Finish the network training/evaluation with learning rate')
#
#

