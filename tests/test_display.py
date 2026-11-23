"""Tests for display's SVG drawing (`_svg`) and `_repr_svg_`.

The three graphs stand in for `protomol.examples` until Step 7 adds it.
"""

import pytest

import protomol as pm
from protomol._display import _svg


def _species() -> pm.StereoMolGraph:
    """(E)-pent-3-en-2-ol, one enantiomer."""
    return pm.from_smiles("C[C@@H](O)/C=C/C")


def _abstraction_ts() -> pm.StereoReactionGraph:
    """C2H6 + OH -> C2H5 + H2O."""
    return pm.ts_graphs(pm.from_smiles("CC.[OH]"), pm.from_smiles("C[CH2].O"))[0]


def _stereoisomerization_ts() -> pm.StereoReactionGraph:
    """(Z) -> (E) but-2-en-1-yl radical, by rotation about C1-C2."""
    z = pm.PlanarBond(atoms=(0, 6, 1, 2, 3, 7), parity=0)
    e = pm.PlanarBond(atoms=(0, 6, 1, 2, 7, 3), parity=0)
    twist = pm.AtropBond(atoms=(0, 6, 1, 2, 3, 7), parity=1)
    return pm.StereoReactionGraph(
        symbols=["C"] * 4 + ["H"] * 7,
        bonds=[
            *[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (1, 6), (2, 7)],
            *[(3, 8), (3, 9), (3, 10)],
        ],
        stereo_changes=[pm.StereoChange(reactant=z, product=e, ts=twist)],
    )


_GRAPHS = {
    "species": _species,
    "abstraction_ts": _abstraction_ts,
    "stereoisomerization_ts": _stereoisomerization_ts,
}


def _is_svg(svg: str) -> bool:
    """Whether `svg` looks like a whole SVG document."""
    return svg.startswith(("<?xml", "<svg")) and "</svg>" in svg


@pytest.mark.parametrize("name", list(_GRAPHS))
def test_svg_renders(name: str) -> None:
    """Each test graph draws as an SVG."""
    assert _is_svg(_svg(_GRAPHS[name]()))


def test_repr_svg_on_every_type() -> None:
    """Each of the four types draws itself in Jupyter, as `_svg` does."""
    smg = _species()
    srg = _abstraction_ts()
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
    gra = _species()
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
