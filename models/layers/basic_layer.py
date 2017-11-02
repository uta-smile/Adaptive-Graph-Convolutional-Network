from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


from AGCN.models.operators import model_operatos as model_ops
from AGCN.models.operators import initializations
import tensorflow as tf


class Layer(object):

    """Abstract base layer class.

    Attributes
    ----------
    name: String, must be unique within a model.
    trainable: Boolean, whether the layer weights
        will be updated during training.
    uses_learning_phase: Whether any operation
        of the layer uses model_ops.in_training_phase()
        or model_ops.in_test_phase().
    input_shape: Shape tuple. Provided for convenience,
      but note that there may be cases in which this
      attribute is ill-defined (e.g. a shared layer
      with multiple input shapes), in which case
      requesting input_shape will raise an Exception.
      Prefer using layer.get_input_shape_for(input_shape),
    output_shape: Shape tuple. See above.
    input, output: Input/output tensor(s). Note that if the layer is used
      more than once (shared layer), this is ill-defined
      and will raise an exception. In such cases, use

    Methods
    -------
    call(x): Where the layer's logic lives.
    __call__(x): Wrapper around the layer logic (`call`).
        If x is a tensor:
            - Connect current layer with last layer from tensor:
            - Add layer to tensor history
        If layer is not built:
    """

    def __init__(self, **kwargs):
        # These properties should have been set
        # by the child class, as appropriate.
        if not hasattr(self, 'uses_learning_phase'):
            self.uses_learning_phase = False

        if not hasattr(self, 'losses'):
            self.losses = []

        # These properties should be set by the user via keyword arguments.
        # note that 'input_dtype', 'input_shape' and 'batch_input_shape'
        # are only applicable to input layers: do not pass these keywords
        # to non-input layers.
        allowed_kwargs = {
            'input_shape', 'batch_input_shape', 'input_dtype', 'name', 'trainable'
        }
        for kwarg in kwargs.keys():
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        name = kwargs.get('name')
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(model_ops.get_uid(prefix))
        self.name = name

        self.trainable = kwargs.get('trainable', True)
        if 'batch_input_shape' in kwargs or 'input_shape' in kwargs:
            # In this case we will create an input layer
            # to insert before the current layer
            if 'batch_input_shape' in kwargs:
                batch_input_shape = tuple(kwargs['batch_input_shape'])
            elif 'input_shape' in kwargs:
                batch_input_shape = (None,) + tuple(kwargs['input_shape'])
            self.batch_input_shape = batch_input_shape
            input_dtype = kwargs.get('input_dtype', tf.float64)
            self.input_dtype = input_dtype

    @staticmethod
    def object_list_uid(object_list):

        def to_list(x):
            """This normalizes a list/tensor into a list.

            If a tensor is passed, we return
            a list of size 1 containing the tensor.
            """
            if isinstance(x, list):
                return x
            return [x]

        object_list = to_list(object_list)
        return ', '.join([str(abs(id(x))) for x in object_list])

    def add_weight(self, shape, initializer, regularizer=None, name=None):
        """Adds a weight variable to the layer.

        Parameters
        ----------
        shape: The shape tuple of the weight.
        initializer: An Initializer instance (callable).
        regularizer: An optional Regularizer instance.
        """
        initializer = initializations.get(initializer)
        weight = initializer(shape, name=name)
        if regularizer is not None:
            self.add_loss(regularizer(weight))

        return weight

    def add_loss(self, losses, inputs=None):
        """Adds losses to model."""
        if losses is None:
            return
        # Update self.losses
        losses = self.to_list(losses)
        if not hasattr(self, 'losses'):
            self.losses = []
        try:
            self.losses += losses
        except AttributeError:
            # In case self.losses isn't settable
            # (i.e. it's a getter method).
            # In that case the `losses` property is
            # auto-computed and shouldn't be set.
            pass
        # Update self._per_input_updates
        if not hasattr(self, '_per_input_losses'):
            self._per_input_losses = {}
        if inputs is not None:
            inputs_hash = self.object_list_uid(inputs)
        else:
            # Updates indexed by None are unconditional
            # rather than input-dependent
            inputs_hash = None
        if inputs_hash not in self._per_input_losses:
            self._per_input_losses[inputs_hash] = []
        self._per_input_losses[inputs_hash] += losses

    def call(self, x):
        """This is where the layer's logic lives.

        Parameters
        ----------
        x: input tensor, or list/tuple of input tensors.

        Returns
        -------
        A tensor or list/tuple of tensors.
        """
        return x

    def __call__(self, x):
        """Wrapper around self.call(), for handling
        Parameters
        ----------
        x: Can be a tensor or list/tuple of tensors.
        """
        return self.call(x)
        # outputs = to_list(self.call(x))
        # return outputs

    @staticmethod
    def to_list(x):
        """This normalizes a list/tensor into a list.

        If a tensor is passed, we return
        a list of size 1 containing the tensor.
        """
        if isinstance(x, list):
            return x
        return [x]

    @staticmethod
    def object_list_uid(object_list):
        object_list = Layer.to_list(object_list)
        return ', '.join([str(abs(id(x))) for x in object_list])