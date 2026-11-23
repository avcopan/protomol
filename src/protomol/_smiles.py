"""`from_smiles`: RDKit parses, stereomolgraph perceives the LS (P7, P16).

protomol then adds the lone-pair LS of sp3 amine N (a Phase 1 approximation of the
low-T scheme), fills in the trivial sites, and checks charge and spin.
"""

from pydantic import ValidationError
from rdkit import Chem

from ._backend import from_smg, to_smg
from ._stereo import Tetrahedral, isomers
from ._types import GraphError, StereoMolGraph, with_stereo
from ._vendor.stereomolgraph.rdmol2graph import RDMol2StereoMolGraph

_RDMOL_TO_SMG = RDMol2StereoMolGraph(
    resonance=True, lone_pair_stereo=True, stereo_complete=False
)
_AMINE_N_NEIGHBORS = 3


def from_smiles(
    smiles: str, charge: int | None = None, spin: int | None = None
) -> StereoMolGraph:
    """The stereo graph of a SMILES, with every atom explicit (stereomol's index rule).

    The atoms are the SMILES atoms in SMILES order, `[H]` atoms included where they are
    written, then the implicit H in the order of the atoms they're bonded to.

    Stereo uses the low-T scheme. Sites whose two configurations give the same graph get
    a fixed parity. Stereo elements the SMILES leaves unspecified stay undetermined
    (parity None) until `stereoisomers`.

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

    try:
        gra = from_smg(_RDMOL_TO_SMG(mol), charge=formal_charge, spin=spin)
        assert isinstance(gra, StereoMolGraph)  # noqa: S101  (narrows the type)
        gra = with_stereo(gra, *_amine_lss(mol, gra))
        return _fill_trivial_sites(gra)
    except ValidationError as err:
        raise GraphError(str(err)) from err


def _amine_lss(mol: Chem.Mol, gra: StereoMolGraph) -> list[Tetrahedral]:
    """A lone-pair `Tetrahedral` on each sp3 N with three neighbours and no LS yet.

    stereomolgraph only gives one when the SMILES has a chiral tag. RDKit calls amide
    and aniline N `SP2`, so they get none. Phase 3 replaces this with the low-T rules.
    """
    has_ls = {ls.atoms[0] for ls in gra.atom_stereo}
    return [
        Tetrahedral(
            atoms=(
                atom.GetIdx(),
                *(neighbor.GetIdx() for neighbor in atom.GetNeighbors()),
                None,
            )
        )
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetDegree() == _AMINE_N_NEIGHBORS
        and atom.GetHybridization() == Chem.HybridizationType.SP3
        and atom.GetIdx() not in has_ls
    ]


def _fill_trivial_sites(gra: StereoMolGraph) -> StereoMolGraph:
    """Give each trivial site the parity of its first isomer.

    A site is trivial when all its configurations give the same graph (stereomolgraph's
    stereo-aware isomorphism), each tested on `gra` with the other sites as they are.
    """
    trivial = []
    for ls in (*gra.atom_stereo, *gra.bond_stereo):
        if ls.parity is not None:
            continue
        isos = isomers(ls)
        smg0, *smgs = (to_smg(with_stereo(gra, iso)) for iso in isos)
        if all(smg == smg0 for smg in smgs):
            trivial.append(isos[0])
    return with_stereo(gra, *trivial)
