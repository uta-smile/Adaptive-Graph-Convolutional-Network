from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from rdkit import Chem

from AGCN.utils.feature import Featurizer
from AGCN.models import MolGraph

"""
constant values for molecular features
"""
possible_atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br',
                      'Fe', 'Ca', 'Cu', 'Mc', 'Pd', 'Pb',
                      'K', 'I', 'Al', 'Ni', 'Mn']

possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [Chem.rdchem.HybridizationType.SP,
                               Chem.rdchem.HybridizationType.SP2,
                               Chem.rdchem.HybridizationType.SP3,
                               Chem.rdchem.HybridizationType.SP3D,
                               Chem.rdchem.HybridizationType.SP3D2]
possible_number_radical_e_list = [0, 1, 2]

reference_lists = [possible_atom_list, possible_numH_list,
                   possible_valence_list, possible_formal_charge_list,
                   possible_number_radical_e_list, possible_hybridization_list]


class ConvMolFeaturizer(Featurizer):

    def __init__(self):
        # Since ConvMol is an object and not a numpy array, need to set dtype to
        # object.
        self.dtype = object
        self.intervals = ConvMolFeaturizer.get_intervals(reference_lists)

    def _featurize(self, mol):
        """Encodes mol as a ConvMol object."""
        # Get the node features
        idx_nodes = [(a.GetIdx(), self.atom_features(a)) for a in mol.GetAtoms()]
        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))
        nodes = np.vstack(nodes)    # only atom feature

        atoms_feature_rd_idx = {}
        bond_feature = {}
        for a in mol.GetAtoms():
            atoms_feature_rd_idx[a.GetIdx()] = self.atom_features(a)
            bond_feature[a.GetIdx()] = []

        b_feature_size = 0
        for b in mol.GetBonds():
            # atom's bond feature, high degree atom has more features
            bond_feature[b.GetBeginAtom().GetIdx()].append(self.bond_features(b))
            bond_feature[b.GetEndAtom().GetIdx()].append(self.bond_features(b))
            b_feature_size = max(self.bond_features(b).shape[0], b_feature_size)

        for a in mol.GetAtoms():
            # sum the bond feature for each atom (high degree may has more bonds)
            bond_feature[a.GetIdx()] = np.sum(bond_feature[a.GetIdx()], axis=0)
            if not isinstance(bond_feature[a.GetIdx()], np.ndarray):
                print("get a atom with no valid bond feature")
                bond_feature[a.GetIdx()] = np.asarray([0] * b_feature_size)
            atoms_feature_rd_idx[a.GetIdx()] = np.hstack((atoms_feature_rd_idx[a.GetIdx()], bond_feature[a.GetIdx()]))

        # Stack nodes into an array
        nodes_bb = np.vstack(list(atoms_feature_rd_idx.values()))

        # Get bond lists with reverse edges included
        edge_list = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        return MolGraph(nodes, canon_adj_list)

    @staticmethod
    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception(
              "input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def get_intervals(l):
        """For list of lists, gets the cumulative products of the lengths"""
        intervals = len(l) * [0]
        # Initialize with 1
        intervals[0] = 1
        for k in range(1, len(l)):
            intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
        return intervals

    @staticmethod
    def safe_index(l, e):
        """Get the index of e in l, providing an index of len(l) if not found"""
        try:
            return l.index(e)
        except "no index e in l":
            return len(l)

    def get_feature_list(self, atom):
        features = 6 * [0]
        features[0] = self.safe_index(possible_atom_list, atom.GetSymbol())
        features[1] = self.safe_index(possible_numH_list, atom.GetTotalNumHs())
        features[2] = self.safe_index(possible_valence_list, atom.GetImplicitValence())
        features[3] = self.safe_index(possible_formal_charge_list, atom.GetFormalCharge())
        features[4] = self.safe_index(possible_number_radical_e_list, atom.GetNumRadicalElectrons())
        features[5] = self.safe_index(possible_hybridization_list, atom.GetHybridization())
        return features

    @staticmethod
    def features_to_id(features, intervals):
        """Convert list of features into index using spacings provided in intervals"""
        id = 0
        for k in range(len(intervals)):
            id += features[k] * intervals[k]

        # Allow 0 index to correspond to null molecule 1
        id += 1
        return id

    @staticmethod
    def id_to_features(id, intervals):
        features = 6 * [0]

        # Correct for null
        id -= 1
        for k in range(0, 6 - 1):
            # print(6-k-1, id)
            features[6 - k - 1] = id // intervals[6 - k - 1]
            id -= features[6 - k - 1] * intervals[6 - k - 1]
        # Correct for last one
        features[0] = id
        return features

    def atom_to_id(self, atom):
        """Return a unique id corresponding to the atom type"""
        features = self.get_feature_list(atom)
        return self.features_to_id(features, self.intervals)

    def atom_features(self, atom, bool_id_feat=False):
        if bool_id_feat:
            return np.array([self.atom_to_id(atom)])
        else:
            return np.array(
                self.one_of_k_encoding_unk(
                    atom.GetSymbol(),
                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                        'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                        'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                        'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
                self.one_of_k_encoding_unk(
                  atom.GetHybridization(),
                  [Chem.rdchem.HybridizationType.SP,
                   Chem.rdchem.HybridizationType.SP2,
                   Chem.rdchem.HybridizationType.SP3,
                   Chem.rdchem.HybridizationType.SP3D,
                   Chem.rdchem.HybridizationType.SP3D2]) +
                [atom.GetIsAromatic()]
            )

    def bond_features(self, bond):
        bt = bond.GetBondType()
        return np.array([
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()]
        )
