"""
include all network models comprised by layers
"""

from AGCN.models.networks.basic_networks import Network
from AGCN.models.networks.basic_AGCN import SimpleAGCN, LongAGCN, MLP_AGCN
from AGCN.models.networks.ResAGCN import ResAGCN, ResAGCNResLap, MLP_ResAGCNResLap
from AGCN.models.networks.DenseAGCN import DenseAGCN, LongDenseAGCN, DenseAGCNResLap, MLP_DenseAGCNResLap
from AGCN.models.networks.Point_AGCN import Point_MLPDenseAGCNResLap, Point_AGCNResLap
from AGCN.models.networks.Seg_AGCN import SegAGCN

# from basic_networks import Network
# from basic_AGCN import SimpleAGCN, LongAGCN, MLP_AGCN
# from ResAGCN import ResAGCN, ResAGCNResLap, MLP_ResAGCNResLap
# from DenseAGCN import DenseAGCN, LongDenseAGCN, DenseAGCNResLap, MLP_DenseAGCNResLap
# from Point_AGCN import Point_MLPDenseAGCNResLap, Point_AGCNResLap
# from Seg_AGCN import SegAGCN
