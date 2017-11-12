from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.operators import initializations, activations
from AGCN.models.layers import Layer
from AGCN.models.layers.graphconv import glorot, truncate_normal


class Merge(Layer):
    """
    fully connected layer,
    input is Tensor (batch_size, n_feat)
    output is Tensor (batch_size, n_classes)

    Maybe used as output layer of classification
    """
    def __init__(self,
                 batch_size,
                 point_num,
                 part_num,
                 **kwargs):
        self.batch_size = batch_size
        self.part_num = part_num    # total number of part in all classes, unique for each part
        self.point_num = point_num      # number of point in each point cloud, here it is the same
        super(Merge, self).__init__(**kwargs)

    def __call__(self, X):
        assert type(X) == list
        X = tf.stack(X)
        X = tf.reshape(X, [self.batch_size, self.point_num, self.part_num])
        return X
