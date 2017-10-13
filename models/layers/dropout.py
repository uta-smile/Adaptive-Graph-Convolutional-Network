from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.layers import Layer


class Dropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Parameters
    ----------
    p: float between 0 and 1. Fraction of the input units to drop.
    seed: A Python integer to use as random seed.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, p, seed=None, **kwargs):
        self.p = p
        self.seed = seed
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        super(Dropout, self).__init__(**kwargs)

    def call(self, x):
        if 0. < self.p < 1.:
            def dropped_inputs():
                retain_prob = 1 - self.p
                return tf.nn.dropout(x * 1., retain_prob, seed=self.seed)

            x = model_ops.in_train_phase(dropped_inputs, lambda: x)
        return x
