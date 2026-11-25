"""Tests for the mirror-image functions and the canonical TS (P50, P60)."""

import random

import autogeom as ag
import pytest

import protomol as pm

from .test_canon import HARD_CASES, _but2ene_rotation_ts, _shuffled

# A union: reverse takes one (P76)
type Graph = pm.AnyGraph
type Reaction = pm.ReactionGraph | pm.StereoReactionGraph


def _chiral_abstraction_ts() -> pm.StereoReactionGraph:
    """OH abstracting a methyl H next to a stereocentre: chiral, in either direction."""
    (ts,) = pm.ts_graphs(
        pm.from_smiles("C[C@H](F)Cl.[OH]"), pm.from_smiles("[CH2][C@H](F)Cl.O")
    )
    return ts


CHIRAL_TS = _chiral_abstraction_ts()


@pytest.mark.parametrize("name", HARD_CASES)
def test_enantiomer_is_an_involution(name: str) -> None:
    """The mirror image of the mirror image is the graph itself, field by field."""
    gra = HARD_CASES[name]
    assert pm.enantiomer(pm.enantiomer(gra)) == gra


@pytest.mark.parametrize("name", HARD_CASES)
def test_enantiomer_keeps_atoms_and_bonds(name: str) -> None:
    """Atom indices are kept: only parities change."""
    gra = HARD_CASES[name]
    mirror = pm.enantiomer(gra)
    assert mirror.symbols == gra.symbols
    assert mirror.bonds == gra.bonds


@pytest.mark.parametrize(
    "gra",
    [
        pm.without_stereo(pm.examples.species()),
        pm.without_stereo(pm.examples.abstraction_ts()),
    ],
)
def test_graphs_without_stereo_are_their_own_mirror_image(gra: Graph) -> None:
    """A MolGraph or ReactionGraph is returned as it is, and is not chiral."""
    assert pm.enantiomer(gra) is gra
    assert not pm.is_chiral(gra)
    assert pm.canonical_enantiomer(gra) is gra


@pytest.mark.parametrize(
    ("smiles", "mirror"),
    [
        ("C[C@H](O)/C=C/C", "C[C@@H](O)/C=C/C"),
        ("O[C@H]1CCCC[C@@H]1O", "O[C@@H]1CCCC[C@H]1O"),
        ("C[N@](F)Cl", "C[N@@](F)Cl"),
    ],
)
def test_enantiomer_inverts_centres(smiles: str, mirror: str) -> None:
    """Each tetrahedral centre is inverted; a planar bond is kept."""
    gra = pm.from_smiles(smiles)
    assert pm.canonical_json(pm.enantiomer(gra)) == pm.canonical_json(
        pm.from_smiles(mirror)
    )


def test_enantiomer_of_a_reflected_geometry() -> None:
    """A geometry reflected through a plane gives the enantiomer's graph."""
    gra = pm.examples.species()
    geo = pm.embed(gra)
    mirror = pm.from_geometry(ag.reflect(geo, [0.0, 0.0, 1.0]))
    assert pm.canonical_json(mirror) == pm.canonical_json(pm.enantiomer(gra))


def test_enantiomer_inverts_change_slots() -> None:
    """The twisted TS slot of a rotation TS is inverted: the other channel (P39)."""
    gra = _but2ene_rotation_ts(1)
    expected = pm.canonical_json(_but2ene_rotation_ts(-1))
    assert pm.canonical_json(pm.enantiomer(gra)) == expected


def test_enantiomer_keeps_undetermined_lss() -> None:
    """An undetermined LS has no mirror image to take."""
    gra = pm.from_smiles("CC(O)CC")
    centre = pm.stereo_at(gra, 1)
    assert centre is not None
    assert centre.parity is None
    assert pm.stereo_at(pm.enantiomer(gra), 1) == centre


@pytest.mark.parametrize(
    ("name", "chiral"),
    [
        ("species", True),
        ("stereoisomerization_ts", True),
        ("but2ene_rotation_ts", True),
        ("meso_butanediol", False),
        ("cis_cyclohexanediol", False),
        ("trans_cyclohexanediol", True),
        ("abstraction_ts", False),
        ("disconnected", False),
    ],
)
def test_is_chiral(name: str, *, chiral: bool) -> None:
    """Meso and planar-only graphs are achiral; one centre or a twisted TS is chiral."""
    assert pm.is_chiral(HARD_CASES[name]) is chiral


def test_is_chiral_ignores_direction() -> None:
    """N inversion's mirror image is its own reverse: one achiral TS (P75)."""
    gra = pm.reaction_graph(pm.from_smiles("C[N@](F)Cl"), pm.from_smiles("C[N@@](F)Cl"))
    assert pm.canonical_json(pm.enantiomer(gra)) != pm.canonical_json(gra)
    assert not pm.is_chiral(gra)
    assert pm.canonical_enantiomer(gra) is gra
    assert pm.is_chiral(CHIRAL_TS)
    assert pm.is_chiral(pm.reverse(CHIRAL_TS))


def test_is_chiral_planar_bond_and_undetermined() -> None:
    """A planar bond alone, or an undetermined centre, is achiral."""
    assert not pm.is_chiral(pm.from_smiles(r"C/C=C/C"))
    assert not pm.is_chiral(pm.from_smiles("CC(O)CC"))
    assert pm.is_chiral(pm.from_smiles("C[C@H](O)CC"))


@pytest.mark.parametrize("name", HARD_CASES)
def test_canonical_enantiomer_is_shared_by_mirror_images(name: str) -> None:
    """A graph and its mirror image, relabelled, pick one canonical JSON."""
    gra = HARD_CASES[name]
    expected = pm.canonical_json(pm.canonical_enantiomer(gra))
    rng = random.Random(name)  # noqa: S311
    for _ in range(10):
        for version in (gra, pm.enantiomer(gra)):
            chosen = pm.canonical_enantiomer(_shuffled(version, rng))
            assert pm.canonical_json(chosen) == expected


@pytest.mark.parametrize("name", HARD_CASES)
def test_canonical_enantiomer_is_the_graph_or_its_mirror_image(name: str) -> None:
    """The result is `gra` or `enantiomer(gra)`, and an achiral graph keeps `gra`."""
    gra = HARD_CASES[name]
    chosen = pm.canonical_enantiomer(gra)
    assert chosen in (gra, pm.enantiomer(gra))
    if not pm.is_chiral(gra):
        assert chosen is gra


def test_canonical_enantiomer_species_takes_the_smaller_json() -> None:
    """For a species the choice is the smaller canonical JSON (like InChI's /m0)."""
    gra = pm.examples.species()
    both = (pm.canonical_json(gra), pm.canonical_json(pm.enantiomer(gra)))
    assert pm.canonical_json(pm.canonical_enantiomer(gra)) == min(both)


@pytest.mark.parametrize(
    "gra", [CHIRAL_TS, pm.examples.stereoisomerization_ts(), _but2ene_rotation_ts(1)]
)
def test_canonical_enantiomer_ignores_direction(gra: Reaction) -> None:
    """A reaction graph and its reverse pick the same mirror image (P60)."""
    chosen = pm.canonical_enantiomer(gra)
    assert pm.canonical_enantiomer(pm.reverse(gra)) == pm.reverse(chosen)


# The canonical TS (P58, P60): canonical_direction(canonical_enantiomer(g))


def _canonical_ts(gra: Reaction) -> str:
    return pm.canonical_json(pm.canonical_direction(pm.canonical_enantiomer(gra))[0])


@pytest.mark.parametrize(
    "gra",
    [
        CHIRAL_TS,
        pm.examples.stereoisomerization_ts(),
        _but2ene_rotation_ts(1),
        pm.examples.abstraction_ts(),
    ],
)
def test_canonical_ts_is_shared_by_all_four_versions(gra: Reaction) -> None:
    """A TS, its reverse, its mirror image and the mirror reversed give one result."""
    versions = (
        gra,
        pm.reverse(gra),
        pm.enantiomer(gra),
        pm.reverse(pm.enantiomer(gra)),
    )
    expected = _canonical_ts(gra)
    rng = random.Random(0)  # noqa: S311
    for version in versions:
        assert _canonical_ts(version) == expected
        shuffled = _shuffled(version, rng)
        assert isinstance(shuffled, pm.ReactionGraph | pm.StereoReactionGraph)
        assert _canonical_ts(shuffled) == expected


def test_canonical_ts_versions_are_distinct() -> None:
    """The four versions of the chiral TS are four different directed graphs."""
    gra = CHIRAL_TS
    versions = (
        gra,
        pm.reverse(gra),
        pm.enantiomer(gra),
        pm.reverse(pm.enantiomer(gra)),
    )
    assert len({pm.canonical_json(v) for v in versions}) == 4


def test_canonical_ts_keeps_atom_indices() -> None:
    """The canonical TS is the input with parities and direction changed, not atoms."""
    chosen, _ = pm.canonical_direction(pm.canonical_enantiomer(CHIRAL_TS))
    assert chosen.symbols == CHIRAL_TS.symbols
    assert chosen.bonds == CHIRAL_TS.bonds
