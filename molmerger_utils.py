"""MolMerger graph construction and featurization utilities."""

from typing import List, Tuple

import numpy as np
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import (
    construct_hydrogen_bonding_info,
    get_atom_chirality_one_hot,
    get_atom_formal_charge,
    get_atom_hydrogen_bonding_one_hot,
    get_atom_hybridization_one_hot,
    get_atom_is_in_aromatic_one_hot,
    get_atom_total_degree_one_hot,
    get_atom_total_num_Hs_one_hot,
    get_atom_type_one_hot,
    get_bond_graph_distance_one_hot,
    get_bond_is_conjugated_one_hot,
    get_bond_is_in_same_ring_one_hot,
    get_bond_stereo_one_hot,
    get_bond_type_one_hot,
)
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from rdkit import Chem
from rdkit.Chem import AllChem


def mol_merger(smiles1: str, smiles2: str) -> str:
    """Merge solute and solvent molecules using the MolMerger charge heuristic."""

    def find_most_charged_atom(mol, partial_charges, is_positive=True):
        charge_function = max if is_positive else min
        charge = charge_function(partial_charges)
        charge_index = partial_charges.index(charge)
        return mol.GetAtomWithIdx(charge_index)

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None:
        raise ValueError(f"Invalid solute SMILES: {smiles1}")
    if mol2 is None:
        raise ValueError(f"Invalid solvent SMILES: {smiles2}")

    AllChem.ComputeGasteigerCharges(mol1)
    AllChem.ComputeGasteigerCharges(mol2)

    gasteiger_charges_mol1 = [
        atom.GetDoubleProp("_GasteigerCharge") for atom in mol1.GetAtoms()
    ]
    gasteiger_charges_mol2 = [
        atom.GetDoubleProp("_GasteigerCharge") for atom in mol2.GetAtoms()
    ]

    most_positive_atom1 = find_most_charged_atom(
        mol1, gasteiger_charges_mol1, is_positive=True
    )
    most_negative_atom1 = find_most_charged_atom(
        mol1, gasteiger_charges_mol1, is_positive=False
    )
    most_positive_atom2 = find_most_charged_atom(
        mol2, gasteiger_charges_mol2, is_positive=True
    )
    most_negative_atom2 = find_most_charged_atom(
        mol2, gasteiger_charges_mol2, is_positive=False
    )

    combined_mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))

    for i, atom in enumerate(combined_mol.GetAtoms()):
        if i < mol1.GetNumAtoms():
            atom.SetDoubleProp("GasteigerChargeFinal", gasteiger_charges_mol1[i])
        else:
            atom.SetDoubleProp(
                "GasteigerChargeFinal",
                gasteiger_charges_mol2[i - mol1.GetNumAtoms()],
            )

    offset = mol1.GetNumAtoms()
    bond_order = Chem.BondType.HYDROGEN
    combined_mol.AddBond(
        most_positive_atom1.GetIdx(),
        most_negative_atom2.GetIdx() + offset,
        order=bond_order,
    )
    combined_mol.AddBond(
        most_negative_atom1.GetIdx(),
        most_positive_atom2.GetIdx() + offset,
        order=bond_order,
    )

    final_combined_mol = Chem.Mol(combined_mol)
    return Chem.MolToSmiles(final_combined_mol)


# Backward-compatible notebook name.
MolMerger = mol_merger


def _construct_atom_feature(
    atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]]
) -> np.ndarray:
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_hs = get_atom_total_num_Hs_one_hot(atom)
    chirality = get_atom_chirality_one_hot(atom)

    return np.concatenate(
        [
            atom_type,
            formal_charge,
            hybridization,
            acceptor_donor,
            aromatic,
            degree,
            total_num_hs,
            chirality,
        ]
    )


def _construct_bond_feature(bond: RDKitBond, dist_matrix) -> np.ndarray:
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    dist = get_bond_graph_distance_one_hot(bond, graph_dist_matrix=dist_matrix)
    return np.concatenate([bond_type, same_ring, conjugated, stereo, dist])


class MolMergerFeaturizer(MolecularFeaturizer):
    """DeepChem featurizer used by the MolMerger notebook."""

    def __init__(self, use_edges: bool = False):
        self.use_edges = use_edges

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        assert datapoint.GetNumAtoms() > 1
        if "mol" in kwargs:
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        dist_matrix = Chem.GetDistanceMatrix(datapoint)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos)
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        src, dest = [], []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        bond_features = None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond, dist_matrix)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features,
        )
