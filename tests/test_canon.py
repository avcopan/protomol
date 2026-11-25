"""Tests for canonical_form and canonical_json.

Chain LS (allenes, cumulenes) are out of scope for now (P45): the back-end's numbering
doesn't see them.
"""

import itertools
import random
from typing import Any, Literal, cast

import pytest

import protomol as pm

# A union: canonical_form, canonical_json and relabel take one (P59)
type Graph = pm.AnyGraph


def _but2ene_rotation_ts(parity: Literal[1, -1]) -> pm.StereoReactionGraph:
    """(Z) → (E) but-2-ene through a twisted TS: an axis whose ends swap by symmetry."""
    g = pm.reaction_graph(pm.from_smiles(r"C/C=C\C"), pm.from_smiles(r"C/C=C/C"))
    (change,) = g.stereo_changes
    z = cast("pm.PlanarBond", change.reactant)
    twist = pm.AtropBond(atoms=z.atoms, parity=parity)
    return pm.with_stereo(
        g, pm.StereoChange(reactant=z, product=change.product, ts=twist)
    )


HARD_CASES: dict[str, Graph] = {
    "species": pm.examples.species(),
    "abstraction_ts": pm.examples.abstraction_ts(),
    "stereoisomerization_ts": pm.examples.stereoisomerization_ts(),
    "meso_butanediol": pm.from_smiles("C[C@@H](O)[C@@H](O)C"),
    "cis_cyclohexanediol": pm.from_smiles("O[C@H]1CCCC[C@H]1O"),
    "trans_cyclohexanediol": pm.from_smiles("O[C@H]1CCCC[C@@H]1O"),
    "but2ene_rotation_ts": _but2ene_rotation_ts(1),
    "disconnected": pm.from_smiles("CC.[OH]"),
    "species_no_stereo": pm.without_stereo(pm.examples.species()),
    "abstraction_ts_no_stereo": pm.without_stereo(pm.examples.abstraction_ts()),
}


def _shuffled(gra: Graph, rng: random.Random) -> Graph:
    atom_map = list(range(len(gra.symbols)))
    rng.shuffle(atom_map)
    return pm.relabel(gra, atom_map)


@pytest.mark.parametrize("name", HARD_CASES)
def test_canonical_json_is_invariant_under_relabelling(name: str) -> None:
    """Verify: 100 random relabellings give the same canonical JSON."""
    gra = HARD_CASES[name]
    expected = pm.canonical_json(gra)
    rng = random.Random(name)  # noqa: S311
    for _ in range(100):
        assert pm.canonical_json(_shuffled(gra, rng)) == expected


@pytest.mark.parametrize("name", HARD_CASES)
def test_atom_map_relabels_to_canonical(name: str) -> None:
    """`canonical == relabel(gra, atom_map)`, and canonicalizing is idempotent."""
    gra = _shuffled(HARD_CASES[name], random.Random(name))  # noqa: S311
    canonical, atom_map = pm.canonical_form(gra)
    assert sorted(atom_map) == list(range(len(gra.symbols)))
    assert canonical == pm.relabel(gra, atom_map)
    assert type(canonical) is type(gra)
    assert pm.canonical_json(gra) == pm.to_json(canonical)
    # Canonical twice is canonical once
    assert pm.canonical_form(canonical)[0] == canonical


def test_equal_smiles_spellings_give_equal_json() -> None:
    """Different SMILES of one stereoisomer give one canonical JSON."""
    # Each group is one stereoisomer (checked with RDKit's canonical SMILES)
    spellings = [
        (  # meso-butane-2,3-diol
            "C[C@@H](O)[C@@H](O)C",
            "C[C@H](O)[C@H](O)C",
            "O[C@@H](C)[C@H](O)C",
            "O[C@H](C)[C@@H](O)C",
        ),
        ("O[C@H]1CCCC[C@H]1O", "O[C@@H]1CCCC[C@@H]1O"),  # cis-cyclohexane-1,2-diol
        ("C[C@@H](O)/C=C/C", "C/C=C/[C@@H](C)O"),  # (R,E)-pent-3-en-2-ol
    ]
    for group in spellings:
        jsons = {pm.canonical_json(pm.from_smiles(s)) for s in group}
        assert len(jsons) == 1, group


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Enantiomers
        (pm.from_smiles("C[C@@H](O)/C=C/C"), pm.from_smiles("C[C@H](O)/C=C/C")),
        (pm.from_smiles("C[C@@H](O)[C@H](O)C"), pm.from_smiles("C[C@H](O)[C@@H](O)C")),
        (pm.from_smiles("O[C@H]1CCCC[C@@H]1O"), pm.from_smiles("O[C@@H]1CCCC[C@H]1O")),
        # Diastereomers
        (pm.from_smiles("C[C@@H](O)[C@H](O)C"), pm.from_smiles("C[C@@H](O)[C@@H](O)C")),
        (pm.from_smiles("O[C@H]1CCCC[C@@H]1O"), pm.from_smiles("O[C@H]1CCCC[C@H]1O")),
        (pm.from_smiles("C[C@@H](O)/C=C/C"), pm.from_smiles(r"C[C@@H](O)/C=C\C")),
        # Determined and undetermined
        (pm.from_smiles("C[C@@H](O)/C=C/C"), pm.from_smiles("CC(O)/C=C/C")),
    ],
)
def test_stereoisomers_differ(first: Graph, second: Graph) -> None:
    """Verify: distinct stereoisomers, enantiomers included, differ."""
    assert pm.canonical_json(first) != pm.canonical_json(second)


def test_symmetric_ts_mirror_images_differ() -> None:
    """The two twisted TSs of a symmetric rotation are mirror images (P39)."""
    first = pm.canonical_json(_but2ene_rotation_ts(1))
    assert first != pm.canonical_json(_but2ene_rotation_ts(-1))


def test_stereoisomer_count_with_two_centres() -> None:
    """CH3CH(OH)CH(OH)CH3: 4 configurations, 3 stereoisomers (R,S ≡ S,R: meso).

    Each configuration is also checked under 20 relabellings.
    """
    base = pm.from_smiles("CC(O)C(O)C")
    sites = [ls for ls in base.atom_stereo if ls.parity is None]
    assert len(sites) == 2
    rng = random.Random(0)  # noqa: S311
    jsons = set()
    for combo in itertools.product(*(pm.isomers(ls) for ls in sites)):
        gra = pm.with_stereo(base, *combo)
        expected = pm.canonical_json(gra)
        for _ in range(20):
            assert pm.canonical_json(_shuffled(gra, rng)) == expected
        jsons.add(expected)
    assert len(jsons) == 3


def test_ts_mirror_images_differ() -> None:
    """The example TS and its mirror image (twist parity -1) differ."""
    ts = pm.examples.stereoisomerization_ts()
    (change,) = ts.stereo_changes
    twist = cast("pm.AtropBond", change.ts)
    mirror = pm.with_stereo(
        ts,
        pm.StereoChange(
            reactant=change.reactant, product=change.product, ts=pm.invert(twist)
        ),
    )
    assert pm.canonical_json(ts) != pm.canonical_json(mirror)


def test_bond_changes_take_part() -> None:
    """A reaction and its reverse differ (breaking and forming swap)."""
    ts = pm.examples.abstraction_ts()
    assert pm.canonical_json(ts) != pm.canonical_json(pm.reverse(ts))
    forward = pm.canonical_json(pm.without_stereo(ts))
    backward = pm.canonical_json(pm.without_stereo(pm.reverse(ts)))
    assert forward != backward


def test_charge_and_spin_are_kept() -> None:
    """Charge and spin carry over to the canonical graph."""
    gra = pm.MolGraph(symbols=["O", "H"], bonds=[(0, 1)], charge=-1)
    canonical, _ = pm.canonical_form(gra)
    assert (canonical.charge, canonical.spin) == (-1, 0)


def test_empty_and_single_atom() -> None:
    """The trivial graphs canonicalize to themselves."""
    empty = pm.MolGraph(symbols=[], bonds=[])
    assert pm.canonical_form(empty) == (empty, ())
    atom = pm.MolGraph(symbols=["He"], bonds=[])
    assert pm.canonical_form(atom) == (atom, (0,))


def test_union_and_precise_arguments_type_check() -> None:
    """A union argument is accepted; a precise one keeps its type (P59, ty checks)."""
    gra: pm.AnyGraph = pm.examples.species()
    canonical, atom_map = pm.canonical_form(gra)
    assert pm.relabel(gra, atom_map) == canonical
    assert pm.canonical_json(gra) == pm.to_json(canonical)
    rxn: pm.ReactionGraph | pm.StereoReactionGraph = pm.examples.abstraction_ts()
    chosen, flipped = pm.canonical_direction(rxn)
    assert chosen == (pm.reverse(pm.examples.abstraction_ts()) if flipped else rxn)
    precise: pm.StereoMolGraph = pm.canonical_form(pm.examples.species())[0]
    relabelled: pm.StereoMolGraph = pm.relabel(pm.examples.species(), atom_map)
    direction: pm.StereoReactionGraph = pm.canonical_direction(
        pm.examples.abstraction_ts()
    )[0]
    assert relabelled == precise
    assert direction == chosen


class TestCanonicalDirection:
    """`canonical_direction` picks the same direction for a TS and its reverse."""

    @pytest.mark.parametrize(
        "name", ["abstraction_ts", "stereoisomerization_ts", "abstraction_ts_no_stereo"]
    )
    def test_reverse_gives_same_result(self, name: str) -> None:
        """G and reverse(g) give one graph, with opposite flags."""
        gra: Any = HARD_CASES[name]  # Any: reverse's type variable rejects a union
        chosen, flipped = pm.canonical_direction(gra)
        again, flipped_again = pm.canonical_direction(pm.reverse(gra))
        assert chosen == again
        assert flipped != flipped_again

    @pytest.mark.parametrize("name", ["abstraction_ts", "but2ene_rotation_ts"])
    def test_flag_and_minimum(self, name: str) -> None:
        """The flag says whether the result is the reverse, and it has the min JSON."""
        gra: Any = HARD_CASES[name]  # Any: reverse's type variable rejects a union
        chosen, flipped = pm.canonical_direction(gra)
        assert chosen == (pm.reverse(gra) if flipped else gra)
        both = (pm.canonical_json(gra), pm.canonical_json(pm.reverse(gra)))
        assert pm.canonical_json(chosen) == min(both)

    def test_relabelled_stable(self) -> None:
        """100 random relabellings, in either direction, give one canonical value."""
        rng = random.Random(0)  # noqa: S311
        gra = pm.examples.abstraction_ts()
        expected = pm.canonical_json(pm.canonical_direction(gra)[0])
        for _ in range(100):
            atom_map = list(range(len(gra.symbols)))
            rng.shuffle(atom_map)
            shuffled = pm.relabel(gra, atom_map)
            if rng.random() < 0.5:
                shuffled = pm.reverse(shuffled)
            assert pm.canonical_json(pm.canonical_direction(shuffled)[0]) == expected

    def test_keeps_atom_indices(self) -> None:
        """The result has the input's atoms and bonds; only the direction differs."""
        gra = pm.examples.abstraction_ts()
        chosen, _ = pm.canonical_direction(gra)
        assert chosen.symbols == gra.symbols
        assert chosen.bonds == gra.bonds

    def test_symmetric_self_reaction(self) -> None:
        """H + H2 → H2 + H is its own reverse: a tie keeps the input, not reversed."""
        gra = pm.ReactionGraph(
            symbols=["H", "H", "H"],
            bonds=[(0, 1), (1, 2)],
            breaking=[(0, 1)],
            forming=[(1, 2)],
            spin=1,
        )
        assert pm.canonical_json(gra) == pm.canonical_json(pm.reverse(gra))
        chosen, flipped = pm.canonical_direction(gra)
        assert chosen == gra
        assert not flipped
        again, _ = pm.canonical_direction(pm.reverse(gra))
        assert pm.canonical_json(again) == pm.canonical_json(chosen)
