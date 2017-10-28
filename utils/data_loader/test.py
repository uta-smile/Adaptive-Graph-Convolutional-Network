"""
test Data Loaders
"""
import os
from collections import namedtuple
from AGCN.utils.data_loader import PointcloudLoader, SMILESLoader, ImageCIFARLoader, ImageFashionLoader
import time


class LoaderTester(object):
    """
    abstract class for tester, include basic functions and members
    """
    def __init__(self, loader_job):
        self.loader_job = loader_job

    def try_load(self):
        raise NotImplementedError


class PointcloudLoaderTester(LoaderTester):

    def try_load(self):

        try:
            loader = PointcloudLoader(self.loader_job.dataset_class,
                                  self.loader_job.dataset_name,
                                  self.loader_job.file_name,
                                  self.loader_job.tasks)
        except 'failed to create instance of PointcloudLoader':
            raise Exception('FAiled')
        assert isinstance(loader, PointcloudLoader)
        print('Loader Object Created!')

        try:
            dataset = loader.load()
        except "loading failed":
            raise Exception('Data loading failed')

        print("Loading Successful!")


class SmilesLoaderTester(LoaderTester):

    def try_load(self):

        try:
            loader = SMILESLoader(self.loader_job.dataset_class,
                                  self.loader_job.dataset_name,
                                  self.loader_job.file_name,
                                  self.loader_job.tasks,
                                  self.loader_job.smiles_filed,
                                  feature='MolGraph',
                                  splitter='index',
                                  download_url=self.loader_job.download_url)
            loader.load()
        except 'failed to create instance of SMILESLoader':
            raise Exception('FAiled')
        assert isinstance(loader, SMILESLoader)


class ImageCIFALoaderTester(LoaderTester):

    def try_load(self):
        try:
            loader = ImageCIFARLoader(
                self.loader_job.dataset_class,
                self.loader_job.dataset_name,
                self.loader_job.file_name,
                self.loader_job.tasks)
        except 'failed to create instance of ImageCIFARLoader':
            raise Exception('FAiled')

        assert isinstance(loader, ImageCIFARLoader)
        start_t = time.time()
        xx = loader.load()
        end_t = time.time() - start_t
        print ("time used for featurization of 50000 32*32 image: %f second", end_t)


class ImageFashionLoaderTester(LoaderTester):

    def try_load(self):
        try:
            loader = ImageFashionLoader(
                self.loader_job.dataset_class,
                self.loader_job.dataset_name,
                self.loader_job.file_name,
                self.loader_job.tasks)
        except 'failed to create instance of ImageFashionLoader':
            raise Exception('FAiled')

        assert isinstance(loader, ImageFashionLoader)
        start_t = time.time()
        loader.featurize(shard_size=2048)
        end_t = time.time() - start_t
        print ("time used for featurization of 60000 28*28 greyscale image: %f second", end_t)


def test_cifarloader():
    cifarimage_tasks = [
        'airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    cifar_loader = namedtuple('cifarimage_loader', 'dataset_class, dataset_name, file_name, tasks')
    cifar_loader.tasks = cifarimage_tasks
    cifar_loader.dataset_class = 'images'
    cifar_loader.dataset_name = 'cifar'
    cifar_loader.file_name = 'cifar10'

    cifar_tester = ImageCIFALoaderTester(cifar_loader)
    try:
        cifar_tester.try_load()
        print("Testing succeed!")
    except "failed":
        raise Exception('FAiled to create ImageCIFARLoader')


def test_fashionloader():
    fashionimage_tasks = [
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    loader = namedtuple('fashionimage_loader', 'dataset_class, dataset_name, file_name, tasks')
    loader.tasks = fashionimage_tasks
    loader.dataset_class = 'images'
    loader.dataset_name = 'fashion'
    loader.file_name = 'fashion'

    tester = ImageFashionLoaderTester(loader)
    try:
        tester.try_load()
        print("Testing succeed!")
    except "failed":
        raise Exception('FAiled to create ImageCIFARLoader')


def test_smiles():
    tox21_tasks = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    smiles_loader_1 = namedtuple('smiles_loader',
                                 'dataset_class, dataset_name, file_name, tasks, smiles_filed, download_url')

    smiles_loader_1.tasks = tox21_tasks
    smiles_loader_1.dataset_class = 'chemistry'
    smiles_loader_1.dataset_name = 'tox21'
    smiles_loader_1.file_name = 'tox21.csv.gz'
    smiles_loader_1.smiles_filed = 'smiles'
    smiles_loader_1.download_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'

    smile_tester = SmilesLoaderTester(smiles_loader_1)
    try:
        smile_tester.try_load()
        print('Testing succeed!')
    except "failed":
        raise Exception("Testing Failed")


def test_pc():

    pc_loader_1 = namedtuple('pc_loader', 'dataset_class, dataset_name, file_name, tasks')

    pc_tasks = ['bicycle', 'pedestrian', 'biker', 'van', 'excavator', 'traffic_sign', 'scooter', 'bench',
                'trash', 'vegetation', 'cyclist', 'umbrella', 'bus', 'ticket_machine', 'trunk', 'post', 'building',
                'traffic_lights', '4wd', 'ute', 'car', 'pillar', 'tree', 'pole', 'truck', 'trailer']

    pc_loader_1.tasks = pc_tasks
    pc_loader_1.dataset_class = 'pointcloud'
    pc_loader_1.dataset_name = 'sydney'
    pc_loader_1.file_name = 'objects'

    pc_loader_test = PointcloudLoaderTester(pc_loader_1)
    try:
        pc_loader_test.try_load()
        print("Testing Succeed!")
    except "failed":
        raise Exception("Testing Failed")


if __name__ =="__main__":
    # test_pc()
    # test_smiles()
    # test_cifarloader()
    test_fashionloader()
