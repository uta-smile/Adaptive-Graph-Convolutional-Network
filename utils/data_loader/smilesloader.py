
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit import Chem

from AGCN.utils.data_loader import DataLoader
from AGCN.utils.save import log, load_csv_files


class SMILESLoader(DataLoader):
    """
        Handles loading of SMILES (.CSV) files.
    """

    def get_shards(self, input_files, shard_size, verbose=True):
        """Defines a generator which returns data for each shard"""

        # TODO extend support of other type of SMILES files, e.g SDF, Excel or local Numpy
        return load_csv_files(input_files, shard_size, verbose=verbose)

    def featurize_shard(self, shard):
        """Featurize a shard of an input dataframe."""
        return self.featurize_smiles_df(shard)

    def featurize_smiles_df(self, shard, log_every=1000, verbose=True):
        """Featurize individual compounds in dataframe.

        Given a featurizer that operates on individual chemical compounds
        or macromolecules, compute & add features for that compound to the
        features dataframe
        """
        field = self.smiles_field
        sample_elems = shard[field].tolist()

        features = []
        for ind, elem in enumerate(sample_elems):
            mol = Chem.MolFromSmiles(elem)
            # TODO (ytz) this is a bandage solution to reorder the atoms so
            # that they're always in the same canonical order. Presumably this
            # should be correctly implemented in the future for graph mols.
            if mol:
                new_order = rdmolfiles.CanonicalRankAtoms(mol)
                mol = rdmolops.RenumberAtoms(mol, new_order)
            if ind % log_every == 0:
                log("Featurizing sample %d" % ind, verbose)
            features.append(self.Featurizer.featurize([mol]))

        valid_inds = np.array([1 if elt.size > 0 else 0 for elt in features], dtype=bool)
        features = [elt for (is_valid, elt) in zip(valid_inds, features) if is_valid]
        return np.squeeze(np.array(features)), valid_inds
