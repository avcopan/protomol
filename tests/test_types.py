"""Tests for the four graph types."""

import itertools

import pytest

import protomol as pm

type AnyGraph = (
    pm.MolGraph | pm.StereoMolGraph | pm.ReactionGraph | pm.StereoReactionGraph
)
TYPES = (pm.MolGraph, pm.StereoMolGraph, pm.ReactionGraph, pm.StereoReactionGraph)


def _build[G: AnyGraph](  # noqa: PLR0913
    cls: type[G],
    symbols: str,
    bonds: list[tuple[int, int]],
    *,
    breaking: tuple[tuple[int, int], ...] = (),
    forming: tuple[tuple[int, int], ...] = (),
    charge: int = 0,
    spin: int = 0,
) -> G:
    """A graph with one-letter symbols (`X` for Cl, `Y` for Br, `I` for I)."""
    names = {"X": "Cl", "Y": "Br"}
    gra = cls(charge=charge, spin=spin)
    for atom, symbol in enumerate(symbols):
        gra.add_node(atom, symbol=names.get(symbol, symbol))
    gra.add_edges_from(bonds)
    gra.add_edges_from(breaking, change="breaking")
    gra.add_edges_from(forming, change="forming")
    return gra


def _chiral[G: AnyGraph](cls: type[G]) -> G:
    """CHFClBr: C0, F1, Cl2, Br3, H4."""
    return _build(cls, "CFXYH", [(0, 1), (0, 2), (0, 3), (0, 4)])


def _sn2[G: pm.ReactionGraph | pm.StereoReactionGraph](cls: type[G]) -> G:
    """I⁻ + CHFClBr → CHFClI + Br⁻: C0, F1, Cl2, Br3, H4, I5."""
    return _build(
        cls,
        "CFXYHI",
        [(0, 1), (0, 2), (0, 4)],
        breaking=((0, 3),),
        forming=((0, 5),),
        charge=-1,
    )


def _allene() -> pm.StereoMolGraph:
    """CH2=C=CH2: C0=C1=C2, H3 and H4 on C0, H5 and H6 on C2, with its axis LS."""
    gra = _build(
        pm.StereoMolGraph, "CCCHHHH", [(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)]
    )
    gra.set_stereo(pm.AtropBond(atoms=(3, 4, 0, 2, 5, 6), parity=1, chain=(1,)))
    return gra


@pytest.mark.parametrize("cls", TYPES)
def test_charge_and_spin_default_to_zero(cls: type[AnyGraph]) -> None:
    """A graph built without a charge or spin has 0 for both (P27)."""
    gra = cls()
    assert (gra.charge, gra.spin) == (0, 0)


def test_spin_parity_is_validated() -> None:
    """CH3 with spin 0 → GraphError (P27); spin 1 is valid."""
    bonds = [(0, 1), (0, 2), (0, 3)]
    with pytest.raises(pm.GraphError, match="parity"):
        _build(pm.MolGraph, "CHHH", bonds).validate()
    _build(pm.MolGraph, "CHHH", bonds, spin=1).validate()
    _build(pm.MolGraph, "CHHH", bonds, charge=1).validate()


def test_spin_must_be_non_negative() -> None:
    """A negative spin is invalid."""
    with pytest.raises(pm.GraphError, match="non-negative"):
        _build(pm.MolGraph, "HH", [(0, 1)], spin=-2).validate()


def test_symbols_must_be_elements() -> None:
    """An unknown element symbol is invalid."""
    gra = _build(pm.MolGraph, "HH", [(0, 1)])
    gra.nodes[1]["symbol"] = "Q"
    with pytest.raises(pm.GraphError, match="Unknown element"):
        gra.validate()


def test_node_data_is_validated() -> None:
    """An atom without a symbol fails the node model."""
    gra = pm.MolGraph()
    gra.add_node(0)
    with pytest.raises(pm.GraphError):
        gra.validate()


def test_nodes_must_be_range_n() -> None:
    """The atoms are 0 to n - 1 (index alignment, P4)."""
    gra = _build(pm.MolGraph, "HH", [(0, 1)])
    gra.remove_node(0)
    gra.add_node(2, symbol="H")
    with pytest.raises(pm.GraphError, match="0 to n - 1"):
        gra.validate()


def test_tetrahedral_neighbours_must_be_bonded() -> None:
    """A species Tetrahedral lists exactly the centre's neighbours."""
    gra = _chiral(pm.StereoMolGraph)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    gra.validate()

    # A neighbour that isn't bonded
    gra.remove_edge(0, 4)
    with pytest.raises(pm.GraphError, match="neighbours"):
        gra.validate()

    # A neighbour that isn't listed (exact for a species)
    gra.add_edge(0, 4)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, None), parity=1))
    with pytest.raises(pm.GraphError, match="neighbours"):
        gra.validate()


def test_ls_atoms_must_be_in_the_graph() -> None:
    """Every atom an LS lists is in the graph."""
    gra = _chiral(pm.StereoMolGraph)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 9), parity=1))
    with pytest.raises(pm.GraphError, match="not in the graph"):
        gra.validate()


def test_chain_must_be_a_path_of_degree_two() -> None:
    """A hand-built allene CH2=C=CH2."""
    gra = _allene()
    gra.validate()

    # An inner atom with a third neighbour
    gra.add_node(7, symbol="H")
    gra.add_edge(1, 7)
    gra.graph["spin"] = 1
    with pytest.raises(pm.GraphError, match="two neighbours"):
        gra.validate()

    # A chain that isn't a path of bonds
    gra = _allene()
    gra.remove_edge(1, 2)
    with pytest.raises(pm.GraphError, match="path"):
        gra.validate()


def test_planar_bond_substituents_must_be_exact() -> None:
    """Ethylene C0=C1: H2, H3 on C0 and H4, H5 on C1."""
    gra = _build(pm.StereoMolGraph, "CCHHHH", [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])
    gra.set_stereo(pm.PlanarBond(atoms=(2, 3, 0, 1, 4, 5), parity=0))
    gra.validate()
    gra.set_stereo(pm.PlanarBond(atoms=(2, None, 0, 1, 4, 5), parity=0))
    with pytest.raises(pm.GraphError, match="neighbours"):
        gra.validate()


def test_ls_must_be_keyed_by_its_site() -> None:
    """A species bond LS lists exactly the substituents of each end."""
    gra = _chiral(pm.StereoMolGraph)
    gra.graph["atom_stereo"][1] = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
    with pytest.raises(pm.GraphError, match="keyed"):
        gra.validate()


def test_reaction_ls_uses_the_bonds_of_its_side() -> None:
    """An unchanged LS holds on both sides; a change's slots on their own side."""
    gra = _sn2(pm.StereoReactionGraph)
    gra.validate()

    # The reacting centre can't be unchanged: its neighbours differ between sides
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    with pytest.raises(pm.GraphError, match="neighbours"):
        gra.validate()
    gra.remove_stereo(0)

    # As a change, each slot uses its side; the TS slot needs only bonded atoms
    gra.set_stereo_change(
        pm.StereoChange(
            reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1),
            product=pm.Tetrahedral(atoms=(0, 1, 2, 4, 5), parity=-1),
            ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
        )
    )
    gra.validate()

    # The forming bond isn't on the reactant side
    gra.set_stereo_change(
        pm.StereoChange(reactant=pm.Tetrahedral(atoms=(0, 1, 2, 4, 5), parity=1))
    )
    with pytest.raises(pm.GraphError, match="neighbours"):
        gra.validate()


def test_copy_does_not_share_stereo_maps() -> None:
    """An LS is keyed by its site."""
    gra = _sn2(pm.StereoReactionGraph)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    gra.set_stereo_change(
        pm.StereoChange(reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    )
    copy = gra.copy()
    assert copy == gra
    assert type(copy) is pm.StereoReactionGraph
    copy.remove_stereo(0)
    assert 0 in gra.atom_stereo
    assert 0 in gra.atom_stereo_changes
    assert copy != gra


def test_types_are_not_subclasses_of_each_other() -> None:
    """A copy has its own LS and change maps."""
    for cls1, cls2 in itertools.permutations(TYPES, 2):
        assert not issubclass(cls1, cls2)


def test_site_is_unchanged_or_changing_not_both() -> None:
    """The four types are separate classes (P20)."""
    gra = _chiral(pm.StereoReactionGraph)
    ls = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
    gra.set_stereo(ls)
    gra.set_stereo_change(pm.StereoChange(reactant=ls, product=pm.invert(ls)))
    with pytest.raises(pm.GraphError, match="both unchanged and changing"):
        gra.validate()


def test_remove_stereo_removes_a_change() -> None:
    """A site is in the plain maps or the change maps, not both."""
    gra = _sn2(pm.StereoReactionGraph)
    gra.set_stereo_change(
        pm.StereoChange(reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    )
    gra.remove_stereo(0)
    assert not gra.atom_stereo_changes
    with pytest.raises(KeyError):
        gra.remove_stereo(0)


def test_stereo_maps_are_read_only() -> None:
    """remove_stereo also removes a changing site."""
    gra = _chiral(pm.StereoMolGraph)
    with pytest.raises(TypeError):
        gra.atom_stereo[0] = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4))  # ty: ignore[invalid-assignment]


def test_equality_is_labelled() -> None:
    """Same type, atoms, bonds, charge, spin and LS (P21); not hashable."""
    assert _chiral(pm.StereoMolGraph) == _chiral(pm.StereoMolGraph)
    assert _chiral(pm.MolGraph) != _chiral(pm.StereoMolGraph)
    assert _build(pm.MolGraph, "CH", [(0, 1)], spin=1) != _build(
        pm.MolGraph, "CH", [(0, 1)], spin=3
    )
    assert _build(pm.MolGraph, "HC", [(0, 1)], spin=1) != _build(
        pm.MolGraph, "CH", [(0, 1)], spin=1
    )
    gra = _chiral(pm.StereoMolGraph)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    assert gra != _chiral(pm.StereoMolGraph)
    with pytest.raises(TypeError):
        hash(gra)


def test_bond_change_defaults_to_none_in_equality() -> None:
    """An edge added without a change equals one added with change=None."""
    gra1 = _build(pm.ReactionGraph, "HH", [(0, 1)])
    gra2 = pm.ReactionGraph()
    gra2.add_nodes_from([(0, {"symbol": "H"}), (1, {"symbol": "H"})])
    gra2.add_edge(0, 1, change=None)
    assert gra1 == gra2


def test_is_determined() -> None:
    """Every LS determined, except that a change's TS slot may be undetermined."""
    gra = _chiral(pm.StereoMolGraph)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4)))
    assert not pm.is_determined(gra)
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    assert pm.is_determined(gra)

    srg = _sn2(pm.StereoReactionGraph)
    srg.set_stereo_change(
        pm.StereoChange(
            reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1),
            ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
        )
    )
    assert pm.is_determined(srg)
    srg.set_stereo_change(
        pm.StereoChange(product=pm.Tetrahedral(atoms=(0, 1, 2, 4, 5)))
    )
    assert not pm.is_determined(srg)


def test_networkx_operations_carry_charge_and_spin() -> None:
    """The LS maps change only through the setters."""
    gra = _build(pm.MolGraph, "CHHH", [(0, 1), (0, 2), (0, 3)], spin=1)
    sub = gra.subgraph([0, 1]).copy()
    assert type(sub) is pm.MolGraph
    assert sub.spin == 1
