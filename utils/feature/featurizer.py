"""
Raw node features extractions for graph object
"""

import numpy as np


class Featurizer(object):
  """
  Abstract class for calculating a set of features for a molecule.

  Child classes implement the _featurize method for calculating features
  for a single molecule.
  """

  def featurize(self, mols, verbose=True, log_every_n=1000):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    mols = list(mols)
    features = []
    for i, mol in enumerate(mols):
      if mol is not None:
        features.append(self._featurize(mol))
      else:
        features.append(np.array([]))

    features = np.asarray(features)
    return features

  def _featurize(self, mol):
    """
    Calculate features for a single molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    """
    return self.featurize(mols)