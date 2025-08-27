"""Multiple RDKit molecules."""

from collections.abc import Iterator, Mapping, Sequence
from typing import TypeVar

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from PIL.Image import Image
from rdkit.Chem import Draw, Mol, rdChemReactions

from . import mol as m_

Mols = tuple[Mol, ...]
AtomKey = tuple[int, int]
AtomMapping = dict[AtomKey, AtomKey]

T = TypeVar("T")


# construct
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


# convert
def image(
    mols: Mols, *, label: bool = True, mapping: AtomMapping | None = None
) -> Image:
    """Generate a display-able image.

    If label=True but no mapping is specified, the flat indices will be used.

    :param mols: RDKit molecules
    :param label: Whether to label the atoms
    :param mapping: An alternative mapping
    :return: PIL Image
    """
    if label or mapping is not None:
        mols = with_flat_index_numbers(mols, mapping=mapping, in_place=False)

    return Draw.MolsToImage(mols)


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


# conversions
def networkx_graph(mols: Mols) -> nx.Graph:
    """Generate NetworkX graph from RDKit molecule object."""
    nx_graph = nx.Graph()

    for pos, mol in enumerate(mols):
        # Add atoms as nodes
        for atom in mol.GetAtoms():
            key = (pos, atom.GetIdx())
            nx_graph.add_node(key, symbol=atom.GetSymbol())

        # Add bonds as edges
        for bond in mol.GetBonds():
            key1 = (pos, bond.GetBeginAtomIdx())
            key2 = (pos, bond.GetEndAtomIdx())
            nx_graph.add_edge(key1, key2)

    return nx_graph


# algorithms
#   - isomorphism
def graph_matcher(mols1: Mols, mols2: Mols) -> GraphMatcher:
    """Determine isomorphism of one molecule onto another."""
    nx_graph1 = networkx_graph(mols1)
    nx_graph2 = networkx_graph(mols2)

    def _node_match(node1, node2) -> bool:
        return node1["symbol"] == node2["symbol"]

    return GraphMatcher(nx_graph1, nx_graph2, node_match=_node_match)


def isomorphic(mols1: Mols, mols2: Mols) -> bool:
    """Determine whether two molecules are isomorphic."""
    matcher = graph_matcher(mols1, mols2)
    return matcher.is_isomorphic()


def isomorphism(mols1: Mols, mols2: Mols) -> AtomMapping | None:
    """Determine all isomorphisms from one molecule to another."""
    return next(isomorphisms_iter(mols1, mols2), None)


def isomorphisms_iter(mols1: Mols, mols2: Mols) -> Iterator[AtomMapping]:
    """Determine all isomorphisms from one molecule to another."""
    matcher = graph_matcher(mols1, mols2)
    return matcher.isomorphisms_iter()


def unique_molecules(mols_lst: Sequence[Mols]) -> list[Mols]:
    """Get unique molecules from a sequence."""
    unique_mols_lst = []
    for mol in mols_lst:
        if not any(isomorphic(mol, m) for m in unique_mols_lst):
            unique_mols_lst.append(mol)
    return unique_mols_lst


#   - reaction enumeration and mapping
def reaction_products_and_mappings(
    reactants: Mols, smarts: str, *, isomorphs: bool = False
) -> tuple[list[Mols], list[AtomMapping]]:
    """Determine possible products and mappings from SMARTS template.

    :param smarts: Reaction SMARTS string
    :param reactants: Reactant molecules
    :param isomorphs: Whether to include isomorphs or filter them out
    :return: Products and reaction mappings
    """
    reaction = rdChemReactions.ReactionFromSmarts(smarts)
    products_lst = list(reaction.RunReactants(reactants))

    # Filter out isomorphs, unless requested to keep them
    products_lst = products_lst if isomorphs else unique_molecules(products_lst)

    # Create a dictionary mapping template numbers to reactant indices
    reactant_index_dct = {}
    for position in range(reaction.GetNumReactantTemplates()):
        template = reaction.GetReactantTemplate(position)
        for atom in template.GetAtoms():
            reactant_index_dct[atom.GetAtomMapNum()] = position

    # Determine the mapping for each product
    mappings = []
    for products in products_lst:
        mapping = {}
        for product_index, mol in enumerate(products):
            for atom in mol.GetAtoms():
                # Determine product atom key
                product_atom_index = atom.GetIdx()
                product_atom_key = (product_index, product_atom_index)
                # Determine reactant atom key
                property_dct = atom.GetPropsAsDict()
                template_number = property_dct.get("old_mapno")
                reactant_index = (
                    reactant_index_dct[template_number]
                    if template_number is not None
                    else product_index
                )
                reactant_atom_index = property_dct["react_atom_idx"]
                reactant_atom_key = (reactant_index, reactant_atom_index)
                mapping[reactant_atom_key] = product_atom_key
        mappings.append(mapping)

    return products_lst, mappings


def reaction_mapping(
    reactants: Mols, products: Mols, smarts: str
) -> AtomMapping | None:
    """Determine reaction mapping based on a SMARTS template.

    :param smarts: Reaction SMARTS string
    :param reactants: Reactant molecules
    :param products: Product molecules
    :return: Reaction mapping
    """
    products_lst_, mappings_ = reaction_products_and_mappings(
        reactants, smarts=smarts, isomorphs=False
    )

    for products_, mapping_ in zip(products_lst_, mappings_, strict=True):
        product_mapping = isomorphism(products_, products)
        if product_mapping is not None:
            return {k_r: product_mapping[k_p_] for k_r, k_p_ in mapping_.items()}

    return None


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
