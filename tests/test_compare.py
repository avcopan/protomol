"""Tests for isomorphism and deduplicate (P48)."""

import random
from typing import Any, Literal

import pytest

import protomol as pm

# Any: ty can't pass a union to the constrained type variables of relabel and
# isomorphism
type Graph = Any


def _shuffled(gra: Graph, seed: int = 0) -> Graph:
    atom_map = list(range(len(gra.symbols)))
    random.Random(seed).shuffle(atom_map)  # noqa: S311
    return pm.relabel(gra, atom_map)


def _mirror_ts() -> pm.StereoReactionGraph:
    """The example stereoisomerization TS with its twist inverted (its mirror image)."""
    ts = pm.examples.stereoisomerization_ts()
    (change,) = ts.stereo_changes
    assert change.ts is not None
    return pm.with_stereo(
        ts,
        pm.StereoChange(
            reactant=change.reactant, product=change.product, ts=pm.invert(change.ts)
        ),
    )


ONE_OF_EACH_TYPE: dict[str, Graph] = {
    "MolGraph": pm.without_stereo(pm.examples.species()),
    "StereoMolGraph": pm.examples.species(),
    "ReactionGraph": pm.without_stereo(pm.examples.abstraction_ts()),
    "StereoReactionGraph": pm.examples.stereoisomerization_ts(),
}

# Pairs of graphs of one type that are not isomorphic
DIFFERENT: dict[str, tuple[Graph, Graph]] = {
    "MolGraph": (
        pm.without_stereo(pm.from_smiles("CCO")),
        pm.without_stereo(pm.from_smiles("COC")),
    ),
    "StereoMolGraph enantiomers": (
        pm.from_smiles("C[C@H](O)F"),
        pm.from_smiles("C[C@@H](O)F"),
    ),
    "StereoMolGraph diastereomers": (
        pm.from_smiles("C/C=C/C"),
        pm.from_smiles(r"C/C=C\C"),
    ),
    "ReactionGraph reverse": (
        pm.without_stereo(pm.examples.abstraction_ts()),
        pm.without_stereo(pm.reverse(pm.examples.abstraction_ts())),
    ),
    "StereoReactionGraph mirror TS": (
        pm.examples.stereoisomerization_ts(),
        _mirror_ts(),
    ),
}


@pytest.mark.parametrize("name", ONE_OF_EACH_TYPE)
def test_isomorphism_relabels_g1_onto_g2(name: str) -> None:
    """`relabel(g1, m) == g2` for a relabelled copy, both ways round."""
    g1 = ONE_OF_EACH_TYPE[name]
    for seed in range(5):
        g2 = _shuffled(g1, seed)
        m = pm.isomorphism(g1, g2)
        assert m is not None
        assert pm.relabel(g1, m) == g2
        back = pm.isomorphism(g2, g1)
        assert back is not None
        assert pm.relabel(g2, back) == g1


@pytest.mark.parametrize("name", DIFFERENT)
def test_isomorphism_is_none_for_different_graphs(name: str) -> None:
    """Stereoisomers, enantiomers included, and a reaction's reverse differ."""
    g1, g2 = DIFFERENT[name]
    assert pm.isomorphism(g1, g2) is None
    assert pm.isomorphism(g1, _shuffled(g2)) is None


def test_symmetric_twisted_ts_differs_from_its_mirror_image() -> None:
    """The two twisted TSs of (Z) → (E) but-2-ene are mirror images (P39, P45)."""
    g = pm.reaction_graph(pm.from_smiles(r"C/C=C\C"), pm.from_smiles("C/C=C/C"))
    (change,) = g.stereo_changes
    z = change.reactant
    assert isinstance(z, pm.PlanarBond)

    def twisted(parity: Literal[1, -1]) -> pm.StereoReactionGraph:
        twist = pm.AtropBond(atoms=z.atoms, parity=parity)
        return pm.with_stereo(
            g, pm.StereoChange(reactant=z, product=change.product, ts=twist)
        )

    assert pm.isomorphism(twisted(1), _shuffled(twisted(1))) is not None
    assert pm.isomorphism(twisted(1), twisted(-1)) is None


def test_isomorphism_compares_charge_and_spin() -> None:
    """Graphs that differ only in charge or spin are not isomorphic."""
    anion = pm.MolGraph(symbols=["O", "H"], bonds=[(0, 1)], charge=-1)
    radical = pm.MolGraph(symbols=["O", "H"], bonds=[(0, 1)])
    quartet = pm.MolGraph(symbols=["O", "H"], bonds=[(0, 1)], spin=3)
    assert pm.isomorphism(anion, radical) is None
    assert pm.isomorphism(radical, quartet) is None


def test_isomorphism_of_empty_graphs_is_empty() -> None:
    """Two empty graphs give `()`, which is not None."""
    empty = pm.MolGraph(symbols=[], bonds=[])
    assert pm.isomorphism(empty, empty) == ()


def test_isomorphism_of_different_sizes_is_none() -> None:
    """Graphs of different sizes are not isomorphic."""
    assert pm.isomorphism(pm.from_smiles("C"), pm.from_smiles("CC")) is None


@pytest.mark.parametrize("name", ONE_OF_EACH_TYPE)
def test_deduplicate_keeps_the_first_of_each_class_in_order(name: str) -> None:
    """Relabelled copies collapse onto the first; distinct graphs are kept in order."""
    gra = ONE_OF_EACH_TYPE[name]
    other = next(
        h
        for pair in DIFFERENT.values()
        for h in pair
        if type(h) is type(gra) and pm.isomorphism(h, gra) is None
    )
    copies = [_shuffled(gra, seed) for seed in range(3)]
    kept = pm.deduplicate([copies[0], other, copies[1], _shuffled(other), copies[2]])
    assert kept == [copies[0], other]


def test_deduplicate_keeps_enantiomers_apart() -> None:
    """Enantiomers are two classes."""
    r, s = DIFFERENT["StereoMolGraph enantiomers"]
    assert pm.deduplicate([r, s, _shuffled(r), _shuffled(s)]) == [r, s]


def test_deduplicate_of_nothing_is_empty() -> None:
    """An empty input gives an empty list."""
    assert pm.deduplicate([]) == []


@pytest.mark.parametrize("function", [pm.isomorphism, lambda *g: pm.deduplicate(g)])
def test_mixed_types_raise(function: Any) -> None:  # noqa: ANN401
    """Graphs of different types are a GraphError."""
    species = pm.examples.species()
    with pytest.raises(pm.GraphError, match="different types"):
        function(species, pm.without_stereo(species))
