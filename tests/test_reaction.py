"""Tests for the reaction functions and relabel."""

from typing import Any

import pytest

import protomol as pm


def _abstraction() -> pm.ReactionGraph:
    """CH3-H + OH -> CH3 + H-OH, aligned by hand: C0, H1 (moving), H2-4, O5, H6."""
    symbols = ["C", "H", "H", "H", "H", "O", "H"]
    reactant = pm.MolGraph(
        symbols=symbols, bonds=[(0, 1), (0, 2), (0, 3), (0, 4), (5, 6)]
    )
    product = pm.MolGraph(
        symbols=symbols, bonds=[(1, 5), (0, 2), (0, 3), (0, 4), (5, 6)]
    )
    return pm.reaction_graph(reactant, product)


def _stereo_reaction() -> pm.StereoReactionGraph:
    """CHFClBr + I- -> CHFClI + Br-, with the centre's LS in both slots."""
    symbols = ["C", "F", "Cl", "Br", "H", "I"]
    reactant = pm.StereoMolGraph(
        symbols=symbols,
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
        charge=-1,
        atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)],
    )
    product = pm.StereoMolGraph(
        symbols=symbols,
        bonds=[(0, 1), (0, 2), (0, 4), (0, 5)],
        charge=-1,
        atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 5, 4), parity=-1)],
    )
    return pm.reaction_graph(reactant, product)


def test_reaction_graph_bond_changes() -> None:
    """Bonds only in the reactant break, bonds only in the product form."""
    gra = _abstraction()
    assert isinstance(gra, pm.ReactionGraph)
    assert gra.breaking == ((0, 1),)
    assert gra.forming == ((1, 5),)
    assert len(gra.bonds) == 6


def test_reaction_graph_stereo_change() -> None:
    """An LS that differs between the sides becomes a StereoChange.

    Br3 leaves and I5 attacks with inversion: the TS slot is the backside TBP.
    """
    gra = _stereo_reaction()
    (change,) = gra.stereo_changes
    assert change.reactant == pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
    assert change.product == pm.Tetrahedral(atoms=(0, 1, 2, 5, 4), parity=-1)
    assert change.ts == pm.TrigonalBipyramidal(atoms=(0, 3, 5, 1, 2, 4), parity=1)
    assert gra.atom_stereo == ()


def test_reaction_graph_keeps_equal_ls() -> None:
    """Equal LS on both sides stay unchanged."""
    g = pm.from_smiles("C[C@@H](O)CC")
    gra = pm.reaction_graph(g, g)
    assert gra.stereo_changes == ()
    assert gra.atom_stereo == g.atom_stereo


@pytest.mark.parametrize("gra", [_abstraction(), _stereo_reaction()])
def test_reactants_products_round_trip(gra: Any) -> None:  # noqa: ANN401 (any graph type)
    """The two sides rebuild the reaction graph."""
    assert pm.reaction_graph(pm.reactants(gra), pm.products(gra)) == gra


@pytest.mark.parametrize("gra", [_abstraction(), _stereo_reaction()])
def test_reverse_twice_is_identity(gra: Any) -> None:  # noqa: ANN401 (any graph type)
    """Reversing swaps the sides, and twice gives the graph back."""
    rev = pm.reverse(gra)
    assert rev != gra
    assert pm.reactants(rev) == pm.products(gra)
    assert pm.products(rev) == pm.reactants(gra)
    assert pm.reverse(rev) == gra


@pytest.mark.parametrize(
    "gra",
    [
        _abstraction(),
        _stereo_reaction(),
        pm.from_smiles("C[C@@H](O)/C=C/C"),
        pm.from_smiles("O"),
    ],
)
def test_relabel_inverse(gra: Any) -> None:  # noqa: ANN401 (any graph type)
    """Relabelling by a permutation and then its inverse is the identity."""
    n = len(gra.symbols)
    perm = [*range(1, n), 0][::-1]
    inverse = [perm.index(k) for k in range(n)]
    new = pm.relabel(gra, perm)
    assert type(new) is type(gra)
    assert pm.relabel(new, inverse) == gra


def test_relabel_direction_matches_atom_map() -> None:
    """Atom k of the result is atom atom_map[k] of the input (the 11-20 meeting)."""
    gra = pm.MolGraph(symbols=["O", "H", "C"], bonds=[(0, 1), (0, 2)])
    new = pm.relabel(gra, [2, 0, 1])
    assert new.symbols == ("C", "O", "H")
    assert new.bonds == ((0, 1), (1, 2))


def test_relabel_moves_stereo() -> None:
    """Atom k of the result is atom atom_map[k] of the input (11-20 meeting)."""
    gra = _stereo_reaction()
    new = pm.relabel(gra, [5, 4, 3, 2, 1, 0])
    (change,) = new.stereo_changes
    assert change.reactant == pm.Tetrahedral(atoms=(5, 4, 3, 2, 1), parity=1)


@pytest.mark.parametrize("atom_map", [[0, 1], [0, 0, 1], [0, 1, 3]])
def test_relabel_rejects_non_permutation(atom_map: list[int]) -> None:
    """Relabelling moves the atoms of LS and changes too."""
    gra = pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)])
    with pytest.raises(ValueError, match="permutation"):
        pm.relabel(gra, atom_map)


def test_without_stereo() -> None:
    """An atom map that is not a permutation raises."""
    gra = _stereo_reaction()
    plain = pm.without_stereo(gra)
    assert isinstance(plain, pm.ReactionGraph)
    assert (plain.bonds, plain.breaking, plain.forming) == (
        gra.bonds,
        gra.breaking,
        gra.forming,
    )
    assert isinstance(pm.without_stereo(pm.reactants(gra)), pm.MolGraph)
    with pytest.raises(TypeError):
        pm.without_stereo(plain)  # ty: ignore[no-matching-overload]


def test_reaction_graph_rejects_misaligned() -> None:
    """Dropping the stereo the LS and keeps the bonds."""
    water = pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)])
    other = pm.MolGraph(symbols=["H", "O", "H"], bonds=[(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="symbols"):
        pm.reaction_graph(water, other)
    triplet = pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1)], spin=2)
    with pytest.raises(ValueError, match="spin"):
        pm.reaction_graph(water, triplet)
    with pytest.raises(TypeError):
        pm.reaction_graph(water, pm.from_smiles("O"))  # ty: ignore[no-matching-overload]
