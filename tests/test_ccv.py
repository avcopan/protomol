"""Tests for ts_graphs, the CCV port (counts from autoccv's tests/test_graph_ts.py)."""

import pytest

import protomol as pm
from protomol._ccv import _CCV, _matcher
from protomol._types import _nx


def _sides(rct: str, prd: str) -> tuple[pm.StereoMolGraph, pm.StereoMolGraph]:
    """The two sides, the products with the reactants' spin (a singlet pair, say)."""
    rct_gra = pm.from_smiles(rct)
    return rct_gra, pm.from_smiles(prd, spin=rct_gra.spin)


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
    """The bonding-score filter keeps two channels."""
    rct_gra, prd_gra = _sides("C1=CC(O[O])CC1", "C1=CC=CC1.O[O]")
    assert len(pm.ts_graphs(rct_gra, prd_gra)) == 2


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


def test_ts_graphs_rejects_different_spin() -> None:
    """Reactants and products with different spins raise."""
    with pytest.raises(ValueError, match="spin"):
        pm.ts_graphs(pm.from_smiles("CCO"), pm.from_smiles("[CH2]O.[CH3]"))
