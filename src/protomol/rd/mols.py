"""Multiple RDKit molecules."""

from collections.abc import Mapping, Sequence
from typing import TypeVar

from rdkit import Chem
from rdkit.Chem import Mol

from . import mol as m_

Mols = tuple[Mol, ...]
AtomKey = tuple[int, int]
AtomMapping = dict[AtomKey, AtomKey]

T = TypeVar("T")


def from_smiles(*smis: str, with_coords: bool = False) -> Mols:
    """Generate RDKit molecules from SMILES.

    :param smis: SMILES strings
    :param with_coords: Whether to add coordinates
    :return: RDKit molecules
    """
    return tuple(m_.from_smiles(smi, with_coords=with_coords) for smi in smis)


# properties
def atom_keys(mols: Mols) -> list[AtomKey]:
    """Get atom keys.

    :param mols: RDKit molecules
    :return: Atom keys
    """
    return [
        (mol_idx, atom_idx)
        for mol_idx, mol in enumerate(mols)
        for atom_idx in range(mol.GetNumAtoms())
    ]


# transformations
def with_flat_index_numbers(
    mols: Mols, mapping: AtomMapping | None = None, in_place: bool = False
) -> Mols:
    """Add flat index atom numbers to RDKit molecules.

    If no mapping is specified, the flat indices will be used.

    :param mols: RDKit molecules
    :param mapping: Atom mapping
    :param in_place: Whether to modify the molecule in place
    :return: RDKit molecule
    """
    mapping = mapping or {k: k for k in atom_keys(mols)}
    num_dcts = atom_keys_split_dict_input(atom_keys_flatten_dict_output(mapping))
    return tuple(
        m_.with_numbers(mol, num_dct=num_dct, in_place=in_place)
        for mol, num_dct in zip(mols, num_dcts, strict=True)
    )


# atom key helpers
def atom_key_molecule_index(key: AtomKey) -> int:
    """Get molecule index of atom key"""
    return key[0]


def atom_key_atom_index(key: AtomKey) -> int:
    """Get molecule index of atom key"""
    return key[1]


def atom_keys_flat_indices(keys: Sequence[AtomKey]) -> list[int]:
    """Flat indices for a complete list of atom keys."""
    # 1. Count occurrences of each molecule index
    mol_idxs = [mol_idx for mol_idx, _ in keys]
    count_dct = {mol_idx: mol_idxs.count(mol_idx) for mol_idx in set(mol_idxs)}
    # 2. Use occurence counts to determine offsets for each atom index
    return [
        atom_idx + sum(count_dct[i] for i in range(mol_idx))
        for mol_idx, atom_idx in keys
    ]


def atom_keys_split_dict_input(
    dct: Mapping[AtomKey, T],
) -> list[dict[int, T]]:
    """Split dict with atom key input into dicts with atom index input.

    :param dct: Dictionary with atom key input
    :return: Dictionaries with atom index input for each molecule
    """
    nmols = max(mol_idx for mol_idx, _ in dct.keys()) + 1
    dcts = []
    for mol_idx in range(nmols):
        dcts.append(
            {
                atom_idx: val
                for (mol_idx_, atom_idx), val in dct.items()
                if mol_idx_ == mol_idx
            }
        )
    return dcts


def atom_keys_flatten_dict_output(dct: Mapping[T, AtomKey]) -> dict[T, int]:
    """Transform dict atom key output into flat index output.

    :param dct: Dictionary with atom key output
    :return: Dictionary with flat index output
    """
    return dict(
        zip(dct.keys(), atom_keys_flat_indices(list(dct.values())), strict=True)
    )
