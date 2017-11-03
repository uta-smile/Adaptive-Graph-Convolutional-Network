"""
hyper parameters preset for each experiments
"""
import os

exp_hps = dict()

exp_hps['test_SimpleAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 10,
    'learning_rate': 0.002,
    'n_filters': 64,
    'l_n_filters': [64, 128, 128, 64],
    'final_feature_n': 256,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/experiments/results'),
    'model_name': 'SimpleAGCN',
    'data_name': 'Tox21'
}

exp_hps['test_LongAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 10,
    'learning_rate': 0.002,
    'n_filters': 128,
    'l_n_filters': [64, 64, 128, 256, 128, 64],
    'final_feature_n': 256,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/experiments/results'),
    'model_name': 'LongAGCN',
    'data_name': 'Tox21'
}

exp_hps['test_ResAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 50,
    'learning_rate': 0.0005,
    'n_filters': 64,
    'final_feature_n': 256,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/experiments/results'),
    'model_name': 'ResAGCN',
    'data_name': 'Tox21'
}

exp_hps['test_LongResAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 50,
    'learning_rate': 0.0005,
    'n_filters': 64,
    'final_feature_n': 256,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/experiments/results'),
    'model_name': 'LongResAGCN',
    'data_name': 'Tox21'
}

exp_hps['test_DenseAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 50,
    'learning_rate': 0.001,
    'n_filters': 64,
    'final_feature_n': 128,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/experiments/results'),
    'model_name': 'DenseAGCN',
    'data_name': 'Tox21'
}

exp_hps['test_LongDenseAGCN'] = {
    'max_hop_K': 2,
    'batch_size': 256,
    'n_epoch': 100,
    'learning_rate': 0.001,
    'n_filters': 64,
    'final_feature_n': 128,
    'seed': 123,
    'n_support': 1,
    'optimizer_beta1': 0.9,
    'optimizer_beta2': 0.999,
    'optimizer_type': 'adam',
    'save_dir': os.path.join(os.environ["HOME"], 'AGCN/AGCN/experiments/results'),
    'model_name': 'LongDenseAGCN',
    'data_name': 'Tox21'
}

exp_hps['gcn_BI'] = {
    'number_hop_max': 2,
    'batch_size': 200,
    'nb_epoch': 50,
    'learning_rate': 0.005,
    'n_filters': 64,
    'n_fully_connected_nodes': 256,
    'seed': 123,
    'n_support': 1,
    'H': 3
}

exp_hps['gcn_BI_reg'] = {
    'number_hop_max': 2,
    'batch_size': 200,
    'nb_epoch': 50,
    'learning_rate': 0.005,
    'n_filters': 64,
    'n_fully_connected_nodes': 256,
    'seed': 123,
    'n_support': 1,
    'H': 3
}
