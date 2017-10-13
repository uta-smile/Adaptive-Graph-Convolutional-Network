"""
define loader for each dataset
"""
from collections import namedtuple

loaderconfig_dict = dict()

""" dataset loader define"""
smiles_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, smiles_filed, download_url')


""" Tox 21 SMILES Dataset """

tox21_tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

smiles_loader.tasks = tox21_tasks
smiles_loader.dataset_class = 'chemistry'
smiles_loader.dataset_name = 'tox21'
smiles_loader.file_name = 'tox21.csv.gz'
smiles_loader.smiles_filed = 'smiles'
smiles_loader.download_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

loaderconfig_dict['tox21'] = smiles_loader

