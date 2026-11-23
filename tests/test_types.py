"""Tests for the four graph types and the functions on them."""

import itertools
import pickle
from typing import Any

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

import protomol as pm

type AnyGraph = (
    pm.MolGraph | pm.StereoMolGraph | pm.ReactionGraph | pm.StereoReactionGraph
)
TYPES = (pm.MolGraph, pm.StereoMolGraph, pm.ReactionGraph, pm.StereoReactionGraph)
NAMES = {"X": "Cl", "Y": "Br"}


def _symbols(letters: str) -> list[str]:
    """One-letter symbols: `X` for Cl, `Y` for Br."""
    return [NAMES.get(letter, letter) for letter in letters]


CHIRAL: dict[str, Any] = {  # CHFClBr
    "symbols": _symbols("CFXYH"),
    "bonds": [(0, 1), (0, 2), (0, 3), (0, 4)],
}
TET = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
SN2: dict[str, Any] = {  # I⁻ + CHFClBr → CHFClI + Br⁻: C0, F1, Cl2, Br3, H4, I5
    "symbols": _symbols("CFXYHI"),
    "bonds": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    "breaking": [(0, 3)],
    "forming": [(0, 5)],
    "charge": -1,
}
ETHYLENE: dict[str, Any] = {  # C0=C1: H2, H3 on C0 and H4, H5 on C1
    "symbols": list("CCHHHH"),
    "bonds": [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
}
PLANAR = pm.PlanarBond(atoms=(2, 3, 0, 1, 4, 5), parity=0)
ALLENE: dict[str, Any] = {  # CH2=C=CH2: C0=C1=C2, H3 and H4 on C0, H5 and H6 on C2
    "symbols": list("CCCHHHH"),
    "bonds": [(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)],
}
ALLENE_LS = pm.AtropBond(atoms=(3, 4, 0, 2, 5, 6), parity=1, chain=(1,))


def _sn2(**stereo: Any) -> pm.StereoReactionGraph:  # noqa: ANN401  any stereo field
    return pm.StereoReactionGraph(**SN2, **stereo)


# Construction and normalization


def test_fields_in_order() -> None:
    """The fields (constructor arguments, JSON fields) and their order (P34)."""
    shared = ["type", "symbols", "bonds", "charge", "spin"]
    assert list(pm.MolGraph.model_fields) == shared
    assert list(pm.StereoMolGraph.model_fields) == [
        *shared,
        "atom_stereo",
        "bond_stereo",
    ]
    assert list(pm.ReactionGraph.model_fields) == [*shared, "breaking", "forming"]
    assert list(pm.StereoReactionGraph.model_fields) == [
        *shared,
        "breaking",
        "forming",
        "atom_stereo",
        "bond_stereo",
        "stereo_changes",
    ]


@pytest.mark.parametrize("cls", TYPES)
def test_type_is_the_class_name(cls: type[AnyGraph]) -> None:
    """`type` defaults to the class name, and no other value is accepted (P31)."""
    assert cls(symbols=["H", "H"], bonds=[(0, 1)]).type == cls.__name__
    with pytest.raises(ValidationError, match="type"):
        cls(type="Graph", symbols=["H", "H"], bonds=[(0, 1)])  # ty: ignore[invalid-argument-type]


def test_arguments_are_keyword_only() -> None:
    """No positional arguments."""
    with pytest.raises(TypeError):
        pm.MolGraph(["H", "H"], [(0, 1)])  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_fields_of_another_type_are_rejected() -> None:
    """extra="forbid": a MolGraph has no `breaking`, a ReactionGraph no stereo."""
    with pytest.raises(ValidationError, match="breaking"):
        pm.MolGraph(**CHIRAL, breaking=[])  # ty: ignore[unknown-argument]
    with pytest.raises(ValidationError, match="atom_stereo"):
        pm.ReactionGraph(**CHIRAL, atom_stereo=[TET])  # ty: ignore[unknown-argument]


def test_constructor_sorts_bonds_and_stereo() -> None:
    """Bonds become sorted pairs i < j in sorted order; LS are sorted by site."""
    g = pm.StereoMolGraph(
        symbols=list("CCFHFH"),
        bonds=[(5, 1), (4, 1), (3, 0), (2, 0), (1, 0)],
        atom_stereo=[
            pm.Tetrahedral(atoms=(1, 0, 4, 5, None)),
            pm.Tetrahedral(atoms=(0, None, 3, 1, 2)),
        ],
        bond_stereo=[pm.PlanarBond(atoms=(4, 5, 1, 0, 3, 2), parity=0)],
    )
    assert g.bonds == ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5))
    assert [pm.site(ls) for ls in g.atom_stereo] == [0, 1]
    assert [pm.site(ls) for ls in g.bond_stereo] == [(0, 1)]
    assert isinstance(g.symbols, tuple)


def test_graphs_are_frozen_and_hashable() -> None:
    """No assignment; equal graphs hash equal and work as dict keys."""
    g = pm.MolGraph(symbols=["H", "H"], bonds=[(0, 1)])
    with pytest.raises(ValidationError, match="frozen"):
        g.spin = 2  # ty: ignore[invalid-assignment]
    assert {g: 1}[pm.MolGraph(symbols=["H", "H"], bonds=[(1, 0)])] == 1


def test_graphs_pickle() -> None:
    """Graphs can be sent between processes."""
    g = _sn2(stereo_changes=[pm.StereoChange(reactant=TET)])
    assert pickle.loads(pickle.dumps(g)) == g  # noqa: S301  our own bytes


def test_numpy_input_coerces_to_int() -> None:
    """A numpy bond array and np.int64 atoms give the same graph as ints (lax mode)."""
    ints = pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET])
    arrays = pm.StereoMolGraph(
        symbols=np.array(CHIRAL["symbols"]),
        bonds=np.array(CHIRAL["bonds"]),
        charge=np.int64(0),  # ty: ignore[invalid-argument-type]  numpy ints coerce
        atom_stereo=[pm.Tetrahedral(atoms=tuple(np.arange(5)), parity=1)],  # ty: ignore[invalid-argument-type]
    )
    assert arrays == ints
    assert hash(arrays) == hash(ints)
    assert type(arrays.bonds[0][0]) is int
    assert type(arrays.charge) is int


def test_fractional_numbers_fail() -> None:
    """A float with a fractional part is not an int."""
    with pytest.raises(ValidationError, match="fractional"):
        pm.MolGraph(symbols=["H"], bonds=[], charge=1.5)
    with pytest.raises(ValidationError, match="fractional"):
        pm.MolGraph(symbols=["H", "H"], bonds=[(0, 0.5)])


def test_errors_are_value_errors() -> None:
    """Constructor errors and GraphError are both ValueError (the portable rule)."""
    with pytest.raises(ValueError, match="Unknown element"):
        pm.MolGraph(symbols=["Q"], bonds=[])
    with pytest.raises(ValueError, match="Invalid JSON"):
        pm.from_json("not json")


@pytest.mark.parametrize("cls", TYPES)
def test_charge_defaults_to_zero_and_spin_to_lowest(cls: type[AnyGraph]) -> None:
    """The default spin is the lowest the electron count allows (P35)."""
    h2 = cls(symbols=["H", "H"], bonds=[(0, 1)])
    assert (h2.charge, h2.spin) == (0, 0)
    ch3: dict[str, Any] = {"symbols": list("CHHH"), "bonds": [(0, 1), (0, 2), (0, 3)]}
    assert cls(**ch3).spin == 1
    assert cls(**ch3, charge=1).spin == 0
    assert cls(symbols=[], bonds=[]).spin == 0


def test_spin_is_validated() -> None:
    """CH3 with spin 0 fails; spin 1 and 3 are valid; negative is invalid."""
    ch3: dict[str, Any] = {"symbols": list("CHHH"), "bonds": [(0, 1), (0, 2), (0, 3)]}
    with pytest.raises(ValidationError, match="parity"):
        pm.MolGraph(**ch3, spin=0)
    assert pm.MolGraph(**ch3, spin=3).spin == 3
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        pm.MolGraph(symbols=["H", "H"], bonds=[(0, 1)], spin=-2)


def test_symbols_must_be_elements() -> None:
    """An unknown symbol, a bare string or a non-string fails."""
    with pytest.raises(ValidationError, match="Unknown element"):
        pm.MolGraph(symbols=["H", "Q"], bonds=[(0, 1)])
    with pytest.raises(ValidationError, match="tuple"):
        pm.MolGraph(symbols="Cl", bonds=[])
    with pytest.raises(ValidationError, match="string"):
        pm.MolGraph(symbols=["H", 1], bonds=[(0, 1)])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("bonds", "match"),
    [
        pytest.param([(0, 3)], "outside", id="atom-out-of-range"),
        pytest.param([(-1, 0)], "negative", id="negative-atom"),
        pytest.param([(1, 1)], "itself", id="self-bond"),
        pytest.param([(0, 1), (1, 0)], "twice", id="bond-twice"),
        pytest.param([(0, 1, 2)], "at most 2", id="triple"),
        pytest.param([1], "tuple", id="not-a-pair"),
    ],
)
def test_bad_bonds_raise(bonds: list[tuple[int, int]], match: str) -> None:
    """A bond is a pair of ints joining two different atoms of the graph, once."""
    with pytest.raises(ValidationError, match=match):
        pm.MolGraph(symbols=["H", "H", "H"], bonds=bonds, spin=1)


def test_breaking_and_forming_are_subsets_of_bonds() -> None:
    """`bonds` is every bond; breaking and forming bonds are among them, not both."""
    g = pm.ReactionGraph(
        symbols=["H", "H", "H"],
        bonds=[(0, 1), (1, 2)],
        breaking=[(0, 1)],
        forming=[(2, 1)],
    )
    assert (g.bonds, g.breaking, g.forming) == (((0, 1), (1, 2)), ((0, 1),), ((1, 2),))
    with pytest.raises(ValidationError, match="must be in `bonds`"):
        pm.ReactionGraph(symbols=["H", "H"], bonds=[], breaking=[(0, 1)])
    with pytest.raises(ValidationError, match="both breaking and forming"):
        pm.ReactionGraph(
            symbols=["H", "H"], bonds=[(0, 1)], breaking=[(0, 1)], forming=[(1, 0)]
        )
    with pytest.raises(ValidationError, match="twice"):
        pm.ReactionGraph(symbols=["H", "H"], bonds=[(0, 1)], forming=[(0, 1), (1, 0)])


def test_types_are_not_subclasses_of_each_other() -> None:
    """The four types are separate classes (P20)."""
    for cls1, cls2 in itertools.permutations(TYPES, 2):
        assert not issubclass(cls1, cls2)


def test_equality_is_field_equality() -> None:
    """Same type, atoms, bonds, charge, spin and LS (P21)."""
    assert pm.StereoMolGraph(**CHIRAL) == pm.StereoMolGraph(**CHIRAL)
    assert pm.MolGraph(**CHIRAL) != pm.StereoMolGraph(**CHIRAL)
    ch: dict[str, Any] = {"symbols": ["C", "H"], "bonds": [(0, 1)]}
    assert pm.MolGraph(**ch, spin=1) != pm.MolGraph(**ch, spin=3)
    assert pm.MolGraph(symbols=["H", "C"], bonds=[(0, 1)]) != pm.MolGraph(**ch)
    assert pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET]) != pm.StereoMolGraph(**CHIRAL)


# LS validation (P30)


def test_tetrahedral_neighbours_must_be_exact() -> None:
    """A species Tetrahedral lists exactly the centre's neighbours."""
    pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET])
    # A neighbour that isn't bonded
    with pytest.raises(ValidationError, match="neighbours"):
        pm.StereoMolGraph(
            symbols=CHIRAL["symbols"], bonds=CHIRAL["bonds"][:-1], atom_stereo=[TET]
        )
    # A neighbour that isn't listed
    with pytest.raises(ValidationError, match="neighbours"):
        pm.StereoMolGraph(
            **CHIRAL, atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 3, None), parity=1)]
        )


def test_ls_atoms_must_be_in_the_graph() -> None:
    """Every atom an LS lists is in the graph."""
    with pytest.raises(ValidationError, match="not in the graph"):
        pm.StereoMolGraph(
            **CHIRAL, atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 3, 9), parity=1)]
        )


def test_chain_must_be_a_path_of_degree_two() -> None:
    """A hand-built allene CH2=C=CH2."""
    pm.StereoMolGraph(**ALLENE, bond_stereo=[ALLENE_LS])
    symbols, bonds = ALLENE["symbols"], ALLENE["bonds"]

    # An inner atom with a third neighbour
    with pytest.raises(ValidationError, match="two neighbours"):
        pm.StereoMolGraph(
            symbols=[*symbols, "H"], bonds=[*bonds, (1, 7)], bond_stereo=[ALLENE_LS]
        )

    # A chain that isn't a path of bonds
    with pytest.raises(ValidationError, match="path"):
        pm.StereoMolGraph(
            symbols=symbols,
            bonds=[b for b in bonds if b != (1, 2)],
            bond_stereo=[ALLENE_LS],
        )


def test_planar_bond_substituents_must_be_exact() -> None:
    """Ethylene's PlanarBond lists every substituent."""
    pm.StereoMolGraph(**ETHYLENE, bond_stereo=[PLANAR])
    with pytest.raises(ValidationError, match="neighbours"):
        pm.StereoMolGraph(
            **ETHYLENE,
            bond_stereo=[pm.PlanarBond(atoms=(2, None, 0, 1, 4, 5), parity=0)],
        )


def test_two_ls_on_one_site_raises() -> None:
    """Each site has at most one LS."""
    with pytest.raises(ValidationError, match="Two LS"):
        pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET, pm.invert(TET)])


def test_stereo_of_the_wrong_type_raises() -> None:
    """Each stereo field takes its own kind of item."""
    with pytest.raises(ValidationError):
        pm.StereoMolGraph(**CHIRAL, atom_stereo=[pm.StereoChange(reactant=TET)])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        pm.StereoMolGraph(**CHIRAL, atom_stereo=[0])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        pm.StereoMolGraph(**ETHYLENE, atom_stereo=[PLANAR])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        pm.StereoMolGraph(**CHIRAL, bond_stereo=[TET])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        _sn2(stereo_changes=[TET])


def test_reaction_ls_uses_the_bonds_of_its_side() -> None:
    """An unchanged LS holds on both sides; a change's slots on their own side."""
    _sn2()

    # The reacting centre can't be unchanged: its neighbours differ between sides
    with pytest.raises(ValidationError, match="neighbours"):
        _sn2(atom_stereo=[TET])

    # As a change, each slot uses its side; the TS slot needs only bonded atoms
    _sn2(
        stereo_changes=[
            pm.StereoChange(
                reactant=TET,
                product=pm.Tetrahedral(atoms=(0, 1, 2, 4, 5), parity=-1),
                ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
            )
        ]
    )

    # The forming bond isn't on the reactant side
    with pytest.raises(ValidationError, match="neighbours"):
        _sn2(
            stereo_changes=[
                pm.StereoChange(
                    reactant=pm.Tetrahedral(atoms=(0, 1, 2, 4, 5), parity=1)
                )
            ]
        )


def test_site_is_unchanged_or_changing_not_both() -> None:
    """A site is in the plain LS or the changes, not both."""
    with pytest.raises(ValidationError, match="both unchanged and changing"):
        pm.StereoReactionGraph(
            **CHIRAL,
            atom_stereo=[TET],
            stereo_changes=[pm.StereoChange(reactant=TET, product=pm.invert(TET))],
        )


def test_two_changes_on_one_site_raises() -> None:
    """Each site changes at most once."""
    change = pm.StereoChange(reactant=TET)
    with pytest.raises(ValidationError, match="Two changes"):
        _sn2(stereo_changes=[change, change])


def test_changes_sort_atom_sites_before_bond_sites() -> None:
    """Atom-site changes by centre, then bond-site changes by end pair (P31)."""
    # CH2=CH-CH3 → CH3-CH=CH2 (made-up, to have one atom and one bond change)
    g = pm.StereoReactionGraph(
        symbols=list("CCCHHHHHH"),
        bonds=[(0, 1), (0, 3), (0, 4), (1, 5), (2, 6), (2, 7), (2, 8), (1, 2)],
        stereo_changes=[
            pm.StereoChange(
                reactant=pm.PlanarBond(atoms=(3, 4, 0, 1, 2, 5), parity=0),
                product=pm.PlanarBond(atoms=(3, 4, 0, 1, 5, 2), parity=0),
            ),
            pm.StereoChange(reactant=pm.Tetrahedral(atoms=(2, 1, 6, 7, 8), parity=1)),
        ],
    )
    assert [type(c.reactant) for c in g.stereo_changes] == [
        pm.Tetrahedral,
        pm.PlanarBond,
    ]


# Functions


def test_stereo_at() -> None:
    """The LS or change at a site, or None; a bond site in either order."""
    ethylene = pm.StereoMolGraph(**ETHYLENE, bond_stereo=[PLANAR])
    assert pm.stereo_at(ethylene, (1, 0)) == PLANAR
    assert pm.stereo_at(ethylene, 0) is None

    change = pm.StereoChange(reactant=TET)
    assert pm.stereo_at(_sn2(stereo_changes=[change]), 0) == change


def test_with_stereo_returns_a_new_graph() -> None:
    """with_stereo adds or replaces the LS at a site, leaving the old graph alone."""
    g = pm.StereoMolGraph(**CHIRAL)
    new = pm.with_stereo(g, TET)
    assert pm.stereo_at(new, 0) == TET
    assert pm.stereo_at(g, 0) is None
    assert pm.stereo_at(pm.with_stereo(new, pm.invert(TET)), 0) == pm.invert(TET)


def test_with_stereo_takes_several_items() -> None:
    """Each item replaces the stereo at its site; two items on one site raise."""
    ethylene = pm.StereoMolGraph(**ETHYLENE)
    tet = pm.Tetrahedral(atoms=(0, 1, 2, 3, None))
    g = pm.with_stereo(ethylene, PLANAR, tet)
    assert (g.atom_stereo, g.bond_stereo) == ((tet,), (PLANAR,))
    assert pm.with_stereo(g) == g
    with pytest.raises(ValidationError, match="Two LS"):
        pm.with_stereo(ethylene, PLANAR, pm.invert(PLANAR))


def test_with_stereo_promotes_graphs_without_stereo() -> None:
    """A MolGraph gives a StereoMolGraph, and a ReactionGraph a StereoReactionGraph."""
    assert pm.with_stereo(pm.MolGraph(**CHIRAL), TET) == pm.StereoMolGraph(
        **CHIRAL, atom_stereo=[TET]
    )
    change = pm.StereoChange(reactant=TET)
    assert pm.with_stereo(pm.ReactionGraph(**SN2), change) == _sn2(
        stereo_changes=[change]
    )


def test_with_stereo_replaces_an_ls_by_a_change() -> None:
    """In a reaction graph, an LS and a change at one site replace each other."""
    g = pm.StereoReactionGraph(**CHIRAL, atom_stereo=[TET])
    change = pm.StereoChange(reactant=TET, product=pm.invert(TET))
    new = pm.with_stereo(g, change)
    assert (new.atom_stereo, new.stereo_changes) == ((), (change,))
    assert pm.with_stereo(new, TET) == g


def test_with_stereo_validates() -> None:
    """The new graph is checked like any other; items of the wrong kind raise."""
    with pytest.raises(ValidationError, match="neighbours"):
        pm.with_stereo(
            pm.StereoMolGraph(**CHIRAL), pm.Tetrahedral(atoms=(0, 1, 2, 3, None))
        )
    with pytest.raises(TypeError, match="takes no StereoChange"):
        pm.with_stereo(pm.MolGraph(**CHIRAL), pm.StereoChange(reactant=TET))  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="Expected an LS or a StereoChange"):
        pm.with_stereo(pm.MolGraph(**CHIRAL), 0)  # ty: ignore[no-matching-overload]


def test_remove_stereo() -> None:
    """remove_stereo removes an LS or a change, and raises KeyError if none."""
    g = pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET])
    assert pm.remove_stereo(g, 0) == pm.StereoMolGraph(**CHIRAL)
    with pytest.raises(KeyError):
        pm.remove_stereo(g, 1)

    ethylene = pm.StereoMolGraph(**ETHYLENE, bond_stereo=[PLANAR])
    assert pm.remove_stereo(ethylene, (1, 0)) == pm.StereoMolGraph(**ETHYLENE)

    srg = _sn2(stereo_changes=[pm.StereoChange(reactant=TET)])
    assert pm.remove_stereo(srg, 0) == _sn2()


def test_neighbors() -> None:
    """Sorted neighbours, through breaking and forming bonds too."""
    assert pm.neighbors(pm.MolGraph(**CHIRAL), 0) == (1, 2, 3, 4)
    assert pm.neighbors(pm.MolGraph(**CHIRAL), 1) == (0,)
    assert pm.neighbors(_sn2(), 0) == (1, 2, 3, 4, 5)
    assert pm.neighbors(_sn2(), 5) == (0,)
    with pytest.raises(IndexError):
        pm.neighbors(pm.MolGraph(**CHIRAL), 5)


def test_nx_is_frozen_and_cached() -> None:
    """The private networkx graph carries the symbols and bond changes."""
    from protomol._types import _nx  # noqa: PLC0415

    nxg = _nx(_sn2())
    assert _nx(_sn2()) is nxg
    assert nxg.nodes[5]["symbol"] == "I"
    assert nxg.edges[0, 3]["change"] == "breaking"
    assert nxg.edges[0, 5]["change"] == "forming"
    assert nxg.edges[0, 1]["change"] is None
    assert nxg.graph == {"charge": -1, "spin": 0}
    with pytest.raises(nx.NetworkXError):
        nxg.add_edge(1, 2)


def test_is_determined() -> None:
    """Every LS determined, except that a change's TS slot may be undetermined."""
    assert not pm.is_determined(
        pm.StereoMolGraph(**CHIRAL, atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 3, 4))])
    )
    assert pm.is_determined(pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET]))

    ts = pm.Tetrahedral(atoms=(0, 1, 2, 4, None))
    assert pm.is_determined(_sn2(stereo_changes=[pm.StereoChange(reactant=TET, ts=ts)]))
    product = pm.Tetrahedral(atoms=(0, 1, 2, 4, 5))
    assert not pm.is_determined(_sn2(stereo_changes=[pm.StereoChange(product=product)]))
