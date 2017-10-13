
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from AGCN.utils.splitter import Splitter
from AGCN.utils.save import log


class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.

    Parameters
    ----------
    include_chirality : : bool, optional (default False)
      Include chirality in scaffolds.
    """
    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.

        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.

        Parameters
        ----------
        mol : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)


class ScaffoldSplitter(Splitter):
    """
    Class for doing data splits based on the scaffold of small molecules.
    """

    def generate_scaffold(self,
                        smiles,
                        include_chirality=False):
        """
        Compute the Bemis-Murcko scaffold for a SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)
        engine = ScaffoldGenerator(include_chirality=include_chirality)
        scaffold = engine.get_scaffold(mol)
        return scaffold

    def split(self,
            dataset,
            frac_train=.5,
            frac_valid=.2,
            frac_test=.3,
            log_every_n=1000,
            verbose=False):
        """
        Splits internal compounds into train/validation/test by scaffold.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)

        scaffolds = {}
        log("About to generate scaffolds", self.verbose)
        data_len = len(dataset)
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                log("Generating scaffold %d/%d" % (ind, data_len), self.verbose)
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds, valid_inds, test_inds = [], [], []
        log("About to sort in scaffold sets", self.verbose)
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
        return train_inds, valid_inds, test_inds
