"""Tests for display's SVG drawing (`_svg`) and `_repr_svg_`."""

import pytest

import protomol as pm
from protomol import examples
from protomol._display import _svg


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
