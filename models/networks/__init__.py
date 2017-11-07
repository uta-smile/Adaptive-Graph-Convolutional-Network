"""
include all network models comprised by layers
"""

from AGCN.models.networks.basic_networks import Network
from AGCN.models.networks.basic_AGCN import SimpleAGCN, LongAGCN, MLP_AGCN
from AGCN.models.networks.ResAGCN import ResAGCN, ResAGCNResLap, MLP_ResAGCNResLap
from AGCN.models.networks.DenseAGCN import DenseAGCN, LongDenseAGCN, DenseAGCNResLap, MLP_DenseAGCNResLap
