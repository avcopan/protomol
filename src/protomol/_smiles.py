"""`from_smiles`: RDKit parses, stereomolgraph perceives the LS (P7, P16).

protomol then puts one chain LS on each cumulated chain (P15), applies the low-T rules
(the pyramidal atoms), fills in the trivial sites, and checks charge and spin.
"""

import itertools

import networkx as nx
from pydantic import ValidationError
from rdkit import Chem

from ._backend import from_smg, to_smg
from ._stereo import AtropBond, PlanarBond, Tetrahedral, isomers
from ._types import (
    GraphError,
    StereoMolGraph,
    _nx,
    neighbors,
    remove_stereo,
    stereo_at,
    with_stereo,
)
from ._vendor.stereomolgraph.rdmol2graph import RDMol2StereoMolGraph

_RDMOL_TO_SMG = RDMol2StereoMolGraph(
    resonance=True, lone_pair_stereo=True, stereo_complete=False
)
_PYRAMIDAL_SYMBOLS = frozenset({"N", "P", "S"})
_PYRAMIDAL_NEIGHBORS = 3
_CHAIN_END_SUBSTITUENTS = 2


def from_smiles(
    smiles: str, charge: int | None = None, spin: int | None = None
) -> StereoMolGraph:
    """The stereo graph of a SMILES, with every atom explicit (stereomol's index rule).

    The atoms are the SMILES atoms in SMILES order, `[H]` atoms included where they are
    written, then the implicit H in the order of the atoms they're bonded to.

    Stereo uses the low-T scheme. Sites whose two configurations give the same graph get
    a fixed parity. Stereo elements the SMILES leaves unspecified stay undetermined
    (parity None) until `stereoisomers`. A cumulated chain gets one LS on its ends, a
    `PlanarBond` or an `AtropBond` with the inner atoms as its `chain`; RDKit doesn't
    read allene stereo, so it is never determined by the SMILES. A double bond or chain
    has stereo only if neither end is sp (linear).

    >>> gra = from_smiles("[H]OC")
    >>> gra.symbols
    ('H', 'O', 'C', 'H', 'H', 'H')
    >>> gra.atom_stereo[0].parity  # C2's Tetrahedral is trivial, so it's filled
    1

    Args:
        smiles: A species SMILES; may have several fragments ("CC.[OH]").
        charge: Checked against the sum of formal charges; default: that sum.
        spin: Unpaired electrons (2S); default: RDKit's radical-electron count
            (maximal pairing, then Hund's rule per atom: [CH2] and [O] → 2; O=O → 0).

    Raises:
        GraphError: For an unparsable SMILES, a reaction SMILES (use `ts_graphs`), a
            charge that differs from the formal charges, or a spin of the wrong parity.
    """
    if ">" in smiles:
        msg = f"{smiles!r} is a reaction SMILES; use `ts_graphs` for a reaction"
        raise GraphError(msg)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        msg = f"RDKit can't parse the SMILES {smiles!r}"
        raise GraphError(msg)
    mol = Chem.AddHs(mol)

    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge is not None and charge != formal_charge:
        msg = (
            f"Charge {charge} differs from the formal charges of {smiles!r} "
            f"({formal_charge})"
        )
        raise GraphError(msg)
    if spin is None:
        spin = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    return _from_rdkit_mol(mol, charge=formal_charge, spin=spin)


def _from_rdkit_mol(mol: Chem.Mol, *, charge: int, spin: int) -> StereoMolGraph:
    """The stereo graph of an RDKit molecule with explicit H, in its atom order.

    The low-T scheme `from_smiles` documents: stereomolgraph perceives the LS from the
    Lewis structure, its resonance structures and the chiral tags, then the chain LS,
    the linear-end rule, the pyramidal atoms and the trivial sites. `from_geometry`
    shares it.
    """
    try:
        gra = from_smg(_RDMOL_TO_SMG(mol), charge=charge, spin=spin)
        assert isinstance(gra, StereoMolGraph)  # noqa: S101  (narrows the type)
        linear = _linear_atoms(mol)
        for chain in _cumulated_chains(mol):
            gra = _with_chain_stereo(gra, chain, linear)
        gra = _without_linear_bond_stereo(gra, linear)
        gra = _with_pyramidal_lss(mol, gra)
        return _fill_trivial_sites(gra)
    except ValidationError as err:
        raise GraphError(str(err)) from err


def _cumulated_chains(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """The maximal paths end1, inner…, end2 of ≥ 2 cumulated double bonds.

    Inner atoms have two neighbours and two double bonds each, in the Kekulé form of
    `mol`. A path runs from its lower-numbered end; the paths are sorted.
    """
    kekule = Chem.Mol(mol)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    double = nx.Graph()
    double.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in kekule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    inner = {
        atom.GetIdx()
        for atom in kekule.GetAtoms()
        if atom.GetDegree() == 2  # noqa: PLR2004
        and double.has_node(atom.GetIdx())
        and double.degree(atom.GetIdx()) == 2  # noqa: PLR2004
    }
    chains = []
    for component in nx.connected_components(double.subgraph(inner)):
        outer = {j for i in component for j in double[i]} - component
        if len(outer) != 2:  # noqa: PLR2004  (a ring of cumulated double bonds)
            continue
        end1, end2 = sorted(outer)
        chains.append(
            tuple(nx.shortest_path(double.subgraph(component | outer), end1, end2))
        )
    return sorted(chains)


def _linear_atoms(mol: Chem.Mol) -> set[int]:
    """The atoms RDKit calls sp in the SMILES's Lewis structure (steric number 2).

    An end of a double bond or a cumulated chain can't carry stereo if it is linear:
    =C=, or the C of a ketene. The test is "not sp" rather than "sp2", because RDKit
    calls a radical centre such as allyl's CH2 sp3, though resonance makes it planar.
    """
    sp = Chem.HybridizationType.SP
    return {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetHybridization() == sp}


def _without_linear_bond_stereo(
    gra: StereoMolGraph, linear: set[int]
) -> StereoMolGraph:
    """`gra` without the bond LS (other than chain LS) that have a linear end."""
    for ls in gra.bond_stereo:
        end1, end2 = ls.atoms[2:4]
        if not ls.chain and {end1, end2} & linear:
            gra = remove_stereo(gra, (end1, end2))
    return gra


def _chain_stereo(
    gra: StereoMolGraph, chain: tuple[int, ...], linear: set[int]
) -> PlanarBond | AtropBond | None:
    """The LS of a chain, or None unless each end is non-linear with a substituent atom.

    As for a double bond, an end's other slot may be a lone pair (`None`): the N of a
    ketenimine R-N=C=CR2, say. A `PlanarBond` for an odd number of double bonds, an
    `AtropBond` for even. The LS is undetermined, since RDKit doesn't read allene or
    cumulene stereo from a SMILES.
    """
    end1, *inner, end2 = chain
    subs = [
        tuple(i for i in neighbors(gra, end) if i != next_)
        for end, next_ in ((end1, chain[1]), (end2, chain[-2]))
    ]
    if {end1, end2} & linear or not all(
        1 <= len(sub) <= _CHAIN_END_SUBSTITUENTS for sub in subs
    ):
        return None
    (a, b), (c, d) = ((*sub, None)[:2] for sub in subs)
    kind = PlanarBond if len(inner) % 2 == 0 else AtropBond
    return kind(atoms=(a, b, end1, end2, c, d), chain=tuple(inner))


def _with_chain_stereo(
    gra: StereoMolGraph, chain: tuple[int, ...], linear: set[int]
) -> StereoMolGraph:
    """`gra` with the per-bond LS on the chain's bonds replaced by the chain LS.

    The chain LS gets its first isomer if the chain is trivial: if either end's two
    substituents are equivalent atoms. "Equivalent" is approximated by equal
    Weisfeiler-Lehman hashes of the stereo-free graph; the exact test comes with the
    canonical form (Phase 3).
    """
    for bond in itertools.pairwise(chain):
        if stereo_at(gra, bond) is not None:
            gra = remove_stereo(gra, bond)
    ls = _chain_stereo(gra, chain, linear)
    if ls is None:
        return gra
    hashes = nx.weisfeiler_lehman_subgraph_hashes(
        _nx(gra), node_attr="symbol", iterations=len(gra.symbols)
    )
    a, b, _, _, c, d = ls.atoms
    if any(
        x is not None and y is not None and hashes[x][-1] == hashes[y][-1]
        for x, y in ((a, b), (c, d))
    ):
        ls = isomers(ls)[0]
    return with_stereo(gra, ls)


def _lone_pairs(atom: Chem.Atom) -> int:
    """The lone pairs of an atom in the molecule's Lewis structure."""
    outer = Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum())
    nonbonding = (
        outer
        - atom.GetFormalCharge()
        - atom.GetTotalValence()
        - atom.GetNumRadicalElectrons()
    )
    return nonbonding // 2


def _is_pi_atom(atom: Chem.Atom) -> bool:
    """Whether an atom is in a π system: a multiple bond or an unpaired electron."""
    return atom.GetNumRadicalElectrons() > 0 or any(
        bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()
    )


def _is_pyramidal(atom: Chem.Atom) -> bool:
    """Whether an atom is a pyramidal centre of the low-T scheme (ch. L1562, L1655).

    Three neighbours and one lone pair (steric number 4), on N, P or S: amines,
    phosphines, sulfoxides and sulfonium ions. The lone pair is counted in the Lewis
    structure, so an onium N (no lone pair) is not one, and a sulfoxide is one in both
    its S=O and S+-O- forms. An N next to a π system (a multiple bond or a radical
    centre) is conjugated and planar, or inverts too fast to keep: amides, anilines,
    enamines, nitrosamines, nitroxides and alpha-amino radicals. P and S stay pyramidal
    when conjugated (barriers ≥ 15 kcal/mol). O+ and C- centres invert too fast and
    are never pyramidal.
    """
    if (
        atom.GetSymbol() not in _PYRAMIDAL_SYMBOLS
        or atom.GetDegree() != _PYRAMIDAL_NEIGHBORS
        or atom.GetIsAromatic()
        or _lone_pairs(atom) != 1
    ):
        return False
    if atom.GetSymbol() == "N":
        return not _is_pi_atom(atom) and not any(
            _is_pi_atom(neighbor) for neighbor in atom.GetNeighbors()
        )
    return True


def _with_pyramidal_lss(mol: Chem.Mol, gra: StereoMolGraph) -> StereoMolGraph:
    """`gra` with a lone-pair `Tetrahedral` on exactly its pyramidal atoms.

    stereomolgraph gives one only for a chiral tag, which is kept (with its parity) on
    a pyramidal atom and dropped elsewhere; each other pyramidal atom gets an
    undetermined one.
    """
    pyramidal = {atom.GetIdx() for atom in mol.GetAtoms() if _is_pyramidal(atom)}
    for ls in gra.atom_stereo:
        if None in ls.atoms and ls.atoms[0] not in pyramidal:
            gra = remove_stereo(gra, ls.atoms[0])
    has_ls = {ls.atoms[0] for ls in gra.atom_stereo}
    return with_stereo(
        gra,
        *(
            Tetrahedral(
                atoms=(
                    i,
                    *(
                        neighbor.GetIdx()
                        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()
                    ),
                    None,
                )
            )
            for i in sorted(pyramidal - has_ls)
        ),
    )


def _fill_trivial_sites(gra: StereoMolGraph) -> StereoMolGraph:
    """Give each trivial site the parity of its first isomer.

    A site is trivial when all its configurations give the same graph (stereomolgraph's
    stereo-aware isomorphism), each tested on `gra` with the other sites as they are.
    stereomolgraph can't hold a chain LS, so the chain LS are left out of the test and
    are never filled here (`_with_chain_stereo` fills the trivial ones).
    """
    chain_free = gra
    for ls in gra.bond_stereo:
        if ls.chain:
            chain_free = remove_stereo(chain_free, ls.atoms[2:4])
    trivial = []
    for ls in (*chain_free.atom_stereo, *chain_free.bond_stereo):
        if ls.parity is not None:
            continue
        isos = isomers(ls)
        smg0, *smgs = (to_smg(with_stereo(chain_free, iso)) for iso in isos)
        if all(smg == smg0 for smg in smgs):
            trivial.append(isos[0])
    return with_stereo(gra, *trivial)
