
import numpy as np

from AGCN.utils.splitter import Splitter


class IndexSplitter(Splitter):
  """
  Class for simple order based splits.
  """
  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=None):
    """
    Splits internal compounds into train/validation/test in provided order.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    num_datapoints = len(dataset)
    train_cutoff = int(frac_train * num_datapoints)
    valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
    indices = range(num_datapoints)
    return (indices[:train_cutoff], indices[train_cutoff:valid_cutoff],
            indices[valid_cutoff:])


class IndiceSplitter(Splitter):
  """
  Class for splits based on input order.
  """

  def __init__(self, verbose=False, valid_indices=None, test_indices=None):
    """
    Parameters
    -----------
    valid_indices: list of int
        indices of samples in the valid set
    test_indices: list of int
        indices of samples in the test set
    """
    self.verbose = verbose
    self.valid_indices = valid_indices
    self.test_indices = test_indices

  def split(self,
            dataset,
            seed=None,
            frac_train=.5,
            frac_valid=.2,
            frac_test=.3,
            log_every_n=None):
    """
    Splits internal compounds into train/validation/test in designated order.
    """
    num_datapoints = len(dataset)
    indices = np.arange(num_datapoints).tolist()
    train_indices = []
    if self.valid_indices is None:
      self.valid_indices = []
    if self.test_indices is None:
      self.test_indices = []
    valid_test = self.valid_indices
    valid_test.extend(self.test_indices)
    for indice in indices:
      if not indice in valid_test:
        train_indices.append(indice)

    return train_indices, self.valid_indices, self.test_indices
