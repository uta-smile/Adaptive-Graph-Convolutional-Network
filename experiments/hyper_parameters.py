"""
hyper parameters preset for each experiments
"""

exp_hps = dict()

exp_hps['test_SimpleAGCN'] = {
    'number_hop_max': 2,
    'batch_size': 256,
    'nb_epoch': 100,
    'learning_rate': 0.001,
    'n_filters': 64,
    'n_fully_connected_nodes': 256,
    'seed': 123,
    'n_support': 1
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

exp_hps['gcn_ll_reg'] = {
    'number_hop_max': 2,
    'batch_size': 200,
    'nb_epoch': 100,
    'learning_rate': 0.005,
    'n_filters': 64,
    'n_fully_connected_nodes': 256,
    'seed': 123,
    'n_support': 1
}

exp_hps['gcn_reg'] = {
    'number_hop_max': 2,
    'batch_size': 200,
    'nb_epoch': 100,
    'learning_rate': 0.005,
    'n_filters': 64,
    'n_fully_connected_nodes': 256,
    'seed': 123,
    'n_support': 1
}

exp_hps['NFP_reg'] = {
    'batch_size': 100,
    'nb_epoch': 100,
    'learning_rate': 0.002,
    'n_filters': 64,
    'FP_length': 1024,
    'seed': 123
}
exp_hps['graphconvreg'] = {
    'batch_size': 128,
    'nb_epoch': 30,
    'learning_rate': 0.0005,
    'n_filters': 128,
    'n_fully_connected_nodes': 256,
    'seed': 123
}


