from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.layers import Layer


class InputLayer(Layer):
    """Layer to be used as an entry point into a graph.

    Create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `input_dtype`).

    Parameters
    ----------
    input_shape: Shape tuple, not including the batch axis.
    batch_input_shape: Shape tuple, including the batch axis.
    input_dtype: Datatype of the input.
    name: Name of the layer (string).
    """

    def __init__(self,
                 input_shape=None,
                 batch_input_shape=None,
                 input_dtype=None,
                 name=None):
        self.uses_learning_phase = False
        self.trainable = False

        if not name:
            prefix = 'input'
            # TODO(rbharath): Keras uses a global var here to maintain
            # unique counts. This seems dangerous. How does tensorflow handle?
            name = prefix + '_' + str(model_ops.get_uid(prefix))
        self.name = name

        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'InputLayer, not both at the same time.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                batch_input_shape = (None,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not input_dtype:
            input_dtype = tf.float32

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

    def __call__(self):
        self.placeholder = tf.placeholder(
            dtype=self.input_dtype, shape=self.batch_input_shape, name=self.name)
        self.placeholder._uses_learning_phase = False
        return [self.placeholder]


def Input(shape=None, batch_shape=None, name=None, dtype=tf.float32):
    """Input() is used to create a placeholder input

    """
    # If batch size not specified
    if len(shape) == 1:
        batch_shape = (None,) + tuple(shape)
    return InputLayer(batch_input_shape=batch_shape, name=name, input_dtype=dtype)


