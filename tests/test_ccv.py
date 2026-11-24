"""Tests for ts_graphs, the CCV port (counts from autoccv's tests/test_graph_ts.py)."""

import pytest

import protomol as pm
from protomol._backend import to_smg
from protomol._ccv import _CCV, _matcher
from protomol._types import _nx, _site


def _sides(rct: str, prd: str) -> tuple[pm.StereoMolGraph, pm.StereoMolGraph]:
    """The two sides, the products with the reactants' spin (a singlet pair, say)."""
    rct_gra = pm.from_smiles(rct)
    return rct_gra, pm.from_smiles(prd, spin=rct_gra.spin)


def _same_stereo_graph(gra1: pm.AnyGraph, gra2: pm.AnyGraph) -> bool:
    """Stereo-aware isomorphism (stereomolgraph's `==`)."""
    return to_smg(gra1) == to_smg(gra2)


def _isomorphic(gra1: pm.AnyGraph, gra2: pm.AnyGraph) -> bool:
    return _matcher(_nx(gra1), _nx(gra2)).is_isomorphic()


@pytest.mark.parametrize(
    ("rct", "prd", "count"),
    [
        ("CCO", "[CH2]O.[CH3]", 1),
        ("CC.[OH]", "C[CH2].O", 1),
        ("CC.[CH3]", "C[CH2].C", 1),
        ("CCO[O]", "[CH2]COO", 1),
        ("CCO[O]", "C=C.O[O]", 1),
    ],
)
def test_ts_graphs_count(rct: str, prd: str, count: int) -> None:
    """The channel counts of autoccv; each channel's sides match the inputs."""
    rct_gra, prd_gra = (pm.without_stereo(g) for g in _sides(rct, prd))
    channels = pm.ts_graphs(rct_gra, prd_gra)
    assert len(channels) == count
    for ts in channels:
        assert isinstance(ts, pm.ReactionGraph)
        assert pm.reactants(ts) == rct_gra
        assert _isomorphic(pm.products(ts), prd_gra)


@pytest.mark.parametrize(
    ("filters", "count"),
    [
        ((True, True, True), 2),
        ((True, True, False), 4),
        ((False, True, True), 4),
        ((False, False, True), 16),
    ],
)
def test_ccv_filters(filters: tuple[bool, bool, bool], count: int) -> None:
    """The counts of autoccv for each combination of the CCV filters."""
    rct_gra, prd_gra = _sides("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]")
    unique_isomorphs, unique_bond_changes, maximum_bond_score = filters
    results = _CCV(rct_gra, prd_gra).results(
        unique_isomorphs=unique_isomorphs,
        unique_bond_changes=unique_bond_changes,
        maximum_bond_score=maximum_bond_score,
    )
    assert len(results) == count


def test_ts_graphs_bonding_score_filter() -> None:
    """The bonding-score filter keeps two channels (four with stereo: syn or anti H)."""
    rct_gra, prd_gra = _sides("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]")
    assert (
        len(pm.ts_graphs(pm.without_stereo(rct_gra), pm.without_stereo(prd_gra))) == 2
    )
    assert len(pm.ts_graphs(rct_gra, prd_gra)) == 4


def test_ts_graphs_with_stereo_keeps_reactant_order() -> None:
    """A stereo channel is in the reactant order, with the LS carried over."""
    rct_gra, prd_gra = _sides("CC.[OH]", "C[CH2].O")
    (ts,) = pm.ts_graphs(rct_gra, prd_gra)
    assert isinstance(ts, pm.StereoReactionGraph)
    assert ts.symbols == rct_gra.symbols
    assert pm.reactants(ts) == rct_gra
    prd = pm.products(ts)
    assert _isomorphic(prd, prd_gra)
    # The C that loses an H loses its LS; the other C keeps its own
    (change,) = ts.stereo_changes
    assert change.product is None
    assert pm.stereo_at(ts, 1) == pm.stereo_at(rct_gra, 1)


def test_ts_graphs_with_stereo_splits_diastereotopic_atoms() -> None:
    """The two H of C3 in (R)-butan-2-ol are diastereotopic: two channels."""
    rct_gra, prd_gra = _sides("C[C@@H](O)CC.[OH]", "C[C@@H](O)[CH]C.O")
    tss = pm.ts_graphs(rct_gra, prd_gra)
    assert len(tss) == 2  # stereo-distinct, each with both sides matching the inputs
    for ts in tss:
        assert _same_stereo_graph(pm.reactants(ts), rct_gra)
        assert _same_stereo_graph(pm.products(ts), prd_gra)
    assert to_smg(tss[0]) != to_smg(tss[1])
    for ts in tss:  # only C3 changes: no CH3 is inverted
        assert [_site(change) for change in ts.stereo_changes] == [3]


def test_ts_graphs_with_stereo_keeps_a_rotation() -> None:
    """(Z) → (E): no bond changes, one channel with the one PlanarBond change."""
    rct_gra, prd_gra = _sides(r"[CH2]/C=C\C", r"[CH2]/C=C/C")
    (ts,) = pm.ts_graphs(rct_gra, prd_gra)
    assert (ts.breaking, ts.forming) == ((), ())
    assert [_site(change) for change in ts.stereo_changes] == [(1, 2)]


def test_stereo_isomorphism_survives_relabelling() -> None:
    """Stereomolgraph matches stereo changes on relabelled atoms (vendored fix)."""
    (ts,) = pm.ts_graphs(*_sides("CC.[OH]", "C[CH2].O"))
    perm = [1, 0, 2, 6, 7, 8, 3, 4, 5, 9]  # swap the two CH3 groups
    assert to_smg(ts) == to_smg(pm.relabel(ts, perm))


def test_ts_graphs_rejects_different_spin() -> None:
    """Reactants and products with different spins raise."""
    with pytest.raises(ValueError, match="spin"):
        pm.ts_graphs(pm.from_smiles("CCO"), pm.from_smiles("[CH2]O.[CH3]"))


def _assert_h_transfer(ts: pm.ReactionGraph | pm.StereoReactionGraph) -> None:
    """One C-H breaks and the same H forms a bond to the other C."""
    ((c1, h1),) = ts.breaking
    ((c2, h2),) = ts.forming
    assert h1 == h2
    assert c1 != c2
    assert {ts.symbols[c1], ts.symbols[c2], ts.symbols[h1]} == {"C", "H"}


def test_ts_graphs_degenerate_h_transfer() -> None:
    """CH4 + CH3 → CH3 + CH4: no empty channel, the H transfer (P44)."""
    rct_gra, prd_gra = _sides("C.[CH3]", "[CH3].C")
    (ts,) = pm.ts_graphs(rct_gra, prd_gra)
    _assert_h_transfer(ts)
    (ts_free,) = pm.ts_graphs(pm.without_stereo(rct_gra), pm.without_stereo(prd_gra))
    _assert_h_transfer(ts_free)


def test_ts_graphs_walden_inversion() -> None:
    """Cl- + CH3CHFCl → inverted: break C-Cl, form Cl-C, with the inversion (P44)."""
    rct_gra, prd_gra = _sides("Cl[C@H](F)C.[Cl-]", "[Cl-].Cl[C@@H](F)C")
    rct_free = pm.without_stereo(rct_gra)
    assert _same_stereo_graph(rct_free, pm.without_stereo(prd_gra))
    assert not _same_stereo_graph(rct_gra, prd_gra)
    (ts,) = pm.ts_graphs(rct_gra, prd_gra)
    assert ts.breaking == ((0, 1),)
    assert ts.forming == ((1, 4),)
    (change,) = ts.stereo_changes
    assert _site(change) == 1
    assert change.reactant is not None
    assert change.product is not None
    assert _same_stereo_graph(pm.reactants(ts), rct_gra)
    assert _same_stereo_graph(pm.products(ts), prd_gra)


def test_ts_graphs_amine_inversion() -> None:
    """An amine N inversion (a lone-pair Tetrahedral) gives one channel (P44)."""
    gra = pm.from_smiles("C[N@](CC)CCC")
    tet = pm.stereo_at(gra, 1)
    assert isinstance(tet, pm.Tetrahedral)
    assert None in tet.atoms
    rct_gra = pm.with_stereo(gra, pm.Tetrahedral(atoms=tet.atoms, parity=1))
    prd_gra = pm.with_stereo(gra, pm.Tetrahedral(atoms=tet.atoms, parity=-1))
    (ts,) = pm.ts_graphs(rct_gra, prd_gra)
    assert (ts.breaking, ts.forming) == ((), ())
    assert [_site(change) for change in ts.stereo_changes] == [1]


def test_ts_graphs_two_rotations_give_no_rotation() -> None:
    """(Z,Z) → (E,E) hexa-2,4-diene: no channel without bond changes (P44).

    Two rotations are two steps. CCV's extra breaks still find double H swaps; this
    unrealistic case doesn't constrain them, so only the rotations are checked.
    """
    rct_gra, prd_gra = _sides(r"C/C=C\C=C/C", r"C/C=C/C=C/C")
    for ts in pm.ts_graphs(rct_gra, prd_gra):
        assert ts.breaking
        assert ts.forming
