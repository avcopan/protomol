import pytest
import rdkit.Chem as Chem  # type: ignore

from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph


def _planar_bonds(smiles: str) -> set[frozenset[int]]:
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    smg = RDMol2StereoMolGraph(resonance=True, lone_pair_stereo=True)(rdmol)
    return {frozenset(bond) for bond in smg.bond_stereo}


@pytest.mark.parametrize(
    ("smiles", "bonds"),
    [
        ("C=C[CH2]", {(0, 1), (1, 2)}),  # allyl
        # vinoxy, from either resonance structure (the cation trick missed the
        # second: an O+ with one bond is not a resonance structure of the cation)
        ("CC=C[O]", {(1, 2)}),
        ("C[CH]C=O", {(1, 2)}),
        # radicals with formal charges (the cation trick was skipped for these)
        ("CC=C[CH][N+](=O)[O-]", {(1, 2), (2, 3), (3, 4)}),
        ("C[CH]C=CC=[O+][O-]", {(1, 2), (2, 3), (3, 4), (4, 5)}),
        # a σ radical doesn't move
        ("C[C]=CC", {(1, 2)}),
    ],
)
def test_radical_resonance(smiles, bonds):
    assert _planar_bonds(smiles) == {frozenset(bond) for bond in bonds}
