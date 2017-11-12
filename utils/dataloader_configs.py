"""
define loader for each dataset
"""
from collections import namedtuple

""" define the configuration of loaders for each dataset """
loaderconfig_dict = dict()

""" SMILES chemical dataset loader declare"""
tox21_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, '
                                           'smiles_filed, split_frac, transformer')
nci_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, smiles_filed, transformer')
delaney_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, smiles_filed, transformer')
clintox_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, smiles_filed, transformer')
sider_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, smiles_filed, '
                                           'transformer')
toxcast_loader = namedtuple('smiles_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, smiles_filed, transformer')

""" Point cloud mesh dataset loader declare"""
sydney_loader = namedtuple('pc_loader', 'dataset_class, dataset_name, file_name, tasks, n_classes, split_frac, transformer')

modelnet40_loader = namedtuple('pc_loader', 'dataset_class, dataset_name, file_name, tasks, n_classes, split_frac, transformer')


""" Image dataset loader declare """
cifar_loader = namedtuple('image_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, transformer')
fashion_loader = namedtuple('image_loader', 'dataset_class, dataset_name, file_name, tasks, split_frac, transformer')


""" Tox21 SMILES Dataset """
tox21_tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

tox21_loader.tasks = tox21_tasks
tox21_loader.dataset_class = 'chemistry'
tox21_loader.dataset_name = 'tox21'
tox21_loader.file_name = 'tox21.csv.gz'
tox21_loader.smiles_filed = 'smiles'
tox21_loader.transformer = 'balancing_w'
loaderconfig_dict['tox21'] = tox21_loader


""" delaney solution SMILES Dataset"""
delaney_tasks = ['measured log solubility in mols per litre']

delaney_loader.tasks = delaney_tasks
delaney_loader.dataset_class = 'chemistry'
delaney_loader.dataset_name = 'delaney'
delaney_loader.file_name = 'delaney-processed.csv'
delaney_loader.smiles_filed = 'smiles'
delaney_loader.transformer = 'normalization_y'
loaderconfig_dict['delaney'] = delaney_loader


""" NCI SMILES Dataset"""
nci_tasks = [
        'CCRF-CEM', 'HL-60(TB)', 'K-562', 'MOLT-4', 'RPMI-8226', 'SR',
        'A549/ATCC', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H23',
        'NCI-H322M', 'NCI-H460', 'NCI-H522', 'COLO 205', 'HCC-2998', 'HCT-116',
        'HCT-15', 'HT29', 'KM12', 'SW-620', 'SF-268', 'SF-295', 'SF-539',
        'SNB-19', 'SNB-75', 'U251', 'LOX IMVI', 'MALME-3M', 'M14', 'MDA-MB-435',
        'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5', 'UACC-257', 'UACC-62', 'IGR-OV1',
        'OVCAR-3', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'NCI/ADR-RES', 'SK-OV-3',
        '786-0', 'A498', 'ACHN', 'CAKI-1', 'RXF 393', 'SN12C', 'TK-10', 'UO-31',
        'PC-3', 'DU-145', 'MCF7', 'MDA-MB-231/ATCC', 'MDA-MB-468', 'HS 578T',
        'BT-549', 'T-47D']

nci_loader.tasks = nci_tasks
nci_loader.dataset_class = 'chemistry'
nci_loader.dataset_name = 'nci'
nci_loader.file_name = 'nci_unique.csv'
nci_loader.smiles_filed = 'smiles'
nci_loader.transformer = 'normalization_y'

loaderconfig_dict['nci'] = nci_loader


""" Clintox SMILES Dataset"""
clintox_loader.tasks = None
clintox_loader.dataset_class = 'chemistry'
clintox_loader.dataset_name = 'clintox'
clintox_loader.file_name = 'clintox.csv.gz'
clintox_loader.smiles_filed = 'smiles'
clintox_loader.transformer = 'balancing_w'

loaderconfig_dict['clintox'] = clintox_loader


""" Sider SMILES Dataset"""
sider_loader.tasks = None
sider_loader.dataset_class = 'chemistry'
sider_loader.dataset_name = 'sider'
sider_loader.file_name = 'sider.csv.gz'
sider_loader.smiles_filed = 'smiles'
sider_loader.transformer = 'normalization_w'
loaderconfig_dict['sider'] = sider_loader


""" Toxcast SMILES Dataset"""
toxcast_loader.tasks = None
toxcast_loader.dataset_class = 'chemistry'
toxcast_loader.dataset_name = 'toxcast'
toxcast_loader.file_name = 'toxcast_data.csv.gz'
toxcast_loader.smiles_filed = 'smiles'
toxcast_loader.transformer = 'balancing_w'

loaderconfig_dict['toxcast'] = toxcast_loader


""" Point Cloud Sydney Dataset"""
# sydney_tasks = ['bicycle', 'pedestrian', 'biker', 'van', 'excavator', 'traffic_sign', 'scooter', 'bench',
#             'trash', 'vegetation', 'cyclist', 'umbrella', 'bus', 'ticket_machine', 'trunk', 'post', 'building',
#             'traffic_lights', '4wd', 'ute', 'car', 'pillar', 'tree', 'pole', 'truck', 'trailer']
sydney_tasks = ['objects']
sydney_loader.tasks = sydney_tasks
sydney_loader.dataset_class = 'pointcloud'
sydney_loader.dataset_name = 'sydney'
sydney_loader.file_name = 'objects'
# sydney_loader.transformer = 'balancing_w'
sydney_loader.transformer = None
sydney_loader.split_frac = [0.8]
sydney_loader.n_classes = 26
loaderconfig_dict['sydney'] = sydney_loader


sydney_tasks = ['objects']
modelnet40_loader.tasks = sydney_tasks
modelnet40_loader.dataset_class = '3Dmesh'
modelnet40_loader.dataset_name = 'modelnet40_ply_hdf5_2048'
modelnet40_loader.transformer = None
modelnet40_loader.split_frac = [0.8]
modelnet40_loader.n_classes = 40
loaderconfig_dict['modelnet40'] = modelnet40_loader

""" Image Fashion MNIST Dataset """
fashionimage_tasks = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

fashion_loader.tasks = fashionimage_tasks
fashion_loader.dataset_class = 'images'
fashion_loader.dataset_name = 'fashion'
fashion_loader.file_name = 'fashion'
fashion_loader.transformer = 'balancing_y'

loaderconfig_dict['fashion'] = fashion_loader


""" Image CIFAR-10 Dataset"""
cifarimage_tasks = [
    'airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

cifar_loader.tasks = cifarimage_tasks
cifar_loader.dataset_class = 'images'
cifar_loader.dataset_name = 'cifar'
cifar_loader.file_name = 'cifar10'
cifar_loader.transformer = 'balancing_y'

loaderconfig_dict['cifar10'] = cifar_loader
