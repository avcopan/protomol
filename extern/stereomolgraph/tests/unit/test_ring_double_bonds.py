import pytest
import rdkit.Chem as Chem  # type: ignore

from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph


def _ring_bond_parity(smiles: str, bond: tuple[int, int]):
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    smg = RDMol2StereoMolGraph(resonance=True, lone_pair_stereo=True)(rdmol)
    return smg.bond_stereo[frozenset(bond)].parity


# A ring of 7 or fewer atoms holds a double bond cis (as InChI, and the
# docstring); from 8 atoms on, it may be trans.
@pytest.mark.parametrize(
    ("smiles", "held"),
    [
        ("C1=CCCCC1", True),  # cyclohexene
        ("C1=CCCCCC1", True),  # cycloheptene
        ("C1=CCCCCCC1", False),  # cyclooctene
    ],
)
def test_ring_double_bond_held_cis(smiles, held):
    parity = _ring_bond_parity(smiles, (0, 1))
    assert (parity is not None) == held
