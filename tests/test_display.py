"""Tests for display's SVG drawing (`_svg`) and `_repr_svg_`."""

from typing import Literal

import numpy as np
import pytest
from rdkit import Chem

import protomol as pm
from protomol import AtropBond, PlanarBond, examples
from protomol._display import _drawing_mol, _svg
from protomol._geometry import _atrop_bond_from_coords, _planar_bond_from_coords


def _is_svg(svg: str) -> bool:
    """Whether `svg` looks like a whole SVG document."""
    return svg.startswith(("<?xml", "<svg")) and "</svg>" in svg


@pytest.mark.parametrize(
    "name", ["species", "abstraction_ts", "stereoisomerization_ts"]
)
def test_svg_renders(name: str) -> None:
    """Each test graph draws as an SVG."""
    assert _is_svg(_svg(getattr(examples, name)()))


def test_repr_svg_on_every_type() -> None:
    """Each of the four types draws itself in Jupyter, as `_svg` does."""
    smg = examples.species()
    srg = examples.abstraction_ts()
    graphs = [
        pm.MolGraph(symbols=smg.symbols, bonds=smg.bonds),
        smg,
        pm.ReactionGraph(
            symbols=srg.symbols,
            bonds=srg.bonds,
            breaking=srg.breaking,
            forming=srg.forming,
        ),
        srg,
    ]
    for gra in graphs:
        assert gra._repr_svg_() == _svg(gra)
        assert _is_svg(gra._repr_svg_())


def test_atom_indices_change_the_svg() -> None:
    """`atom_indices` adds the indices to the drawing."""
    gra = examples.species()
    assert _svg(gra, atom_indices=True) != _svg(gra)


def test_undetermined_atom_ls_is_marked() -> None:
    """An undetermined atom LS gets a "?" note; a determined one doesn't."""
    # RDKit draws an atom note (here only the "?") as paths of class 'note'
    assert "class='note'" in _svg(pm.from_smiles("CC(O)CC"))  # C1 undetermined
    assert "class='note'" not in _svg(pm.from_smiles("C[C@@H](O)CC"))


def test_charge_reaches_bond_order_perception() -> None:
    """The graph's charge goes to bond-order perception."""
    acetate = pm.from_smiles("CC(=O)[O-]")
    # the same graph given charge 0 perceives no C=O (and a radical O)
    neutral = pm.StereoMolGraph(
        symbols=acetate.symbols,
        bonds=acetate.bonds,
        atom_stereo=acetate.atom_stereo,
    )
    assert _svg(acetate) != _svg(neutral)


# The drawing's stereo: RDKit reads the wedges and the layout back as the input's


def _drawn(gra: pm.StereoMolGraph | pm.StereoReactionGraph) -> Chem.Mol:
    mol, _, _ = _drawing_mol(gra, atom_indices=False)
    return mol


def _cip(mol: Chem.Mol) -> dict[object, str]:
    """RDKit's R/S and E/Z labels of `mol`, by atom and by bond ends."""
    labels: dict[object, str] = {
        a.GetIdx(): a.GetProp("_CIPCode")
        for a in mol.GetAtoms()
        if a.HasProp("_CIPCode")
    }
    for b in mol.GetBonds():
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            labels[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = str(b.GetStereo())
    return labels


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@@H](O)CC",
        "C[C@H](O)CC",
        "F[C@](Cl)(Br)I",
        "C[S@](=O)CC",  # a lone pair
        "C[S@@](=O)CC",
        "N[C@@H](C)C(=O)O",
        "C[C@@H]1CC[C@H]1O",
        "C/C=C/C",
        r"C/C=C\C",
        r"C[C@@H](O)/C=C\C",
        r"F/C=C/C=C\Cl",
    ],
)
def test_drawn_stereo_matches_the_smiles(smiles: str) -> None:
    """RDKit's R/S and E/Z from the drawing's wedges and 2D layout are the SMILES's."""
    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))  # from_smiles's atom order
    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    mol = Chem.Mol(_drawn(pm.from_smiles(smiles)))
    Chem.SanitizeMol(mol)
    Chem.AssignChiralTypesFromBondDirs(mol)
    Chem.DetectBondStereochemistry(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    assert _cip(ref)
    assert _cip(mol) == _cip(ref)


def _lifted_coords(mol: Chem.Mol) -> np.ndarray:
    """The 2D coordinates with each wedge's far atom lifted (a hash's lowered)."""
    xyz = np.array(mol.GetConformer().GetPositions())
    for bond in mol.GetBonds():
        direction = bond.GetBondDir()
        if direction in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            up = direction == Chem.BondDir.BEGINWEDGE
            xyz[bond.GetEndAtomIdx(), 2] = 1.0 if up else -1.0
    return xyz


@pytest.mark.parametrize("parity", [1, -1])
def test_allene_axis_wedge(parity: Literal[1, -1]) -> None:
    """A determined allene axis gets one mark, which reads back as its parity."""
    gra = pm.from_smiles("CC=C=CC")
    (undetermined,) = gra.bond_stereo
    assert isinstance(undetermined, AtropBond)
    axis = AtropBond(atoms=undetermined.atoms, parity=parity, chain=undetermined.chain)
    mol = _drawn(pm.with_stereo(gra, axis))
    marked = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBondDir() != Chem.BondDir.NONE
    ]
    assert len(marked) == 1  # the methyls are unmarked without atom indices
    assert marked[0][0] in axis.atoms[2:4]
    assert _atrop_bond_from_coords(axis, _lifted_coords(mol)) == axis


@pytest.mark.parametrize("parity_swap", [False, True])
def test_cumulene_layout(parity_swap: bool) -> None:  # noqa: FBT001
    """A determined chain `PlanarBond` (butatriene) is laid out cis or trans."""
    gra = pm.from_smiles("CC=C=C=CC")
    (ls,) = gra.bond_stereo
    a, b, end1, end2, c, d = ls.atoms
    atoms = (a, b, end1, end2, d, c) if parity_swap else ls.atoms
    ls = PlanarBond(atoms=atoms, parity=0, chain=ls.chain)
    mol = _drawn(pm.with_stereo(gra, ls))
    assert _planar_bond_from_coords(ls, mol.GetConformer().GetPositions()) == ls


def test_radical_planar_bonds_laid_out() -> None:
    """A radical's planar bonds (no double bond for RDKit) are laid out as given."""
    gra = examples.stereoisomerization_ts()
    mol = _drawn(gra)
    xyz = mol.GetConformer().GetPositions()
    lss = [*gra.bond_stereo, *(c.reactant for c in gra.stereo_changes if c.reactant)]
    for ls in lss:
        if isinstance(ls, PlanarBond):
            assert _planar_bond_from_coords(ls, xyz) == ls


def test_twin_terminal_sites_marked_only_with_indices() -> None:
    """A CH3 carbon's wedge shows which numbered H is where, so only with indices."""
    gra = pm.from_smiles("CC")

    def wedges(*, atom_indices: bool) -> int:
        mol, _, _ = _drawing_mol(gra, atom_indices=atom_indices)
        return sum(b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds())

    assert wedges(atom_indices=False) == 0
    assert wedges(atom_indices=True) == 2


def test_undetermined_bond_ls_is_crossed() -> None:
    """An undetermined `PlanarBond` on a double bond crosses it."""
    mol = _drawn(pm.from_smiles("CC=CC"))
    assert mol.GetBondBetweenAtoms(1, 2).GetStereo() == Chem.BondStereo.STEREOANY


@pytest.mark.parametrize(
    ("smiles", "bond"), [("CC=C=CC", (2, 3)), ("CC=C=C=CC", (2, 3))]
)
def test_undetermined_chain_ls_has_a_note(smiles: str, bond: tuple[int, int]) -> None:
    """An undetermined chain LS puts a "?" on its middle bond (RDKit can't cross it)."""
    mol = _drawn(pm.from_smiles(smiles))
    assert mol.GetBondBetweenAtoms(*bond).GetProp("bondNote") == "?"


def test_reaction_colours() -> None:
    """Breaking red, forming blue and dashed, a changing centre magenta.

    No wedge goes on a breaking or forming bond.
    """
    gra = examples.abstraction_ts()
    mol, bond_colors, atom_colors = _drawing_mol(gra, atom_indices=False)
    assert {bond_colors[b] for b in gra.breaking} == {(1.0, 0.0, 0.0)}
    assert {bond_colors[b] for b in gra.forming} == {(0.0, 0.0, 1.0)}
    for i, j in gra.forming:
        assert mol.GetBondBetweenAtoms(i, j).GetBondType() == Chem.BondType.HYDROGEN
    for change in gra.stereo_changes:
        assert isinstance(change.reactant, pm.Tetrahedral)
        assert atom_colors[change.reactant.atoms[0]] == (1.0, 0.0, 1.0)
    for i, j in (*gra.breaking, *gra.forming):
        assert mol.GetBondBetweenAtoms(i, j).GetBondDir() == Chem.BondDir.NONE


def test_bond_stereo_change_is_magenta() -> None:
    """A bond whose stereo changes is magenta."""
    gra = examples.stereoisomerization_ts()
    _, bond_colors, _ = _drawing_mol(gra, atom_indices=False)
    (planar,) = [
        c.reactant for c in gra.stereo_changes if isinstance(c.reactant, PlanarBond)
    ]
    _, _, end1, end2, _, _ = planar.atoms
    assert bond_colors[min(end1, end2), max(end1, end2)] == (1.0, 0.0, 1.0)


def test_radical_and_charges_are_drawn() -> None:
    """The Lewis structure's radicals and formal charges reach the drawing."""
    allyl = _drawn(pm.from_smiles("C=C[CH2]"))
    assert sum(a.GetNumRadicalElectrons() for a in allyl.GetAtoms()) == 1
    nitro = _drawn(pm.from_smiles("C[N+](=O)[O-]"))
    assert sorted(
        a.GetFormalCharge() for a in nitro.GetAtoms() if a.GetFormalCharge()
    ) == [-1, 1]
