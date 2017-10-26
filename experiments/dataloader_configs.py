"""
define loader for each dataset
"""
from collections import namedtuple

loaderconfig_dict = dict()

""" dataset loader define"""
smiles_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, smiles_filed, download_url')
pc_loader = namedtuple('pc_loader', 'dataset_class, dataset_name, file_name, tasks')


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


""" Point Cloud Sydney urban Dataset"""
pc_tasks = ['bicycle', 'pedestrian', 'biker', 'van', 'excavator', 'traffic_sign', 'scooter', 'bench',
            'trash', 'vegetation', 'cyclist', 'umbrella', 'bus', 'ticket_machine', 'trunk', 'post', 'building',
            'traffic_lights', '4wd', 'ute', 'car', 'pillar', 'tree', 'pole', 'truck', 'trailer']

pc_loader.tasks = pc_tasks
pc_loader.dataset_class = 'pointcloud'
pc_loader.dataset_name = 'sydney'
pc_loader.file_name = 'objects'

loaderconfig_dict['sydney'] = pc_loader
