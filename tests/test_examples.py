"""Tests for the test graphs in `protomol.examples`, against their golden JSON."""

from pathlib import Path

import pytest

import protomol as pm
from protomol import examples

_DATA = Path(__file__).parent / "data" / "examples"
_NAMES = ["species", "abstraction_ts", "stereoisomerization_ts"]


@pytest.mark.parametrize("name", _NAMES)
def test_matches_golden_json(name: str) -> None:
    """Each example's JSON matches its golden file (tests/data/examples/<name>.json)."""
    golden = (_DATA / f"{name}.json").read_text().rstrip("\n")
    assert pm.to_json(getattr(examples, name)()) == golden


@pytest.mark.parametrize("name", _NAMES)
def test_is_workflow_ready(name: str) -> None:
    """Every LS of each example is determined."""
    assert pm.is_determined(getattr(examples, name)())


def test_species() -> None:
    """(E)-pent-3-en-2-ol: a chiral centre at C1 and an E bond at C3=C4."""
    gra = examples.species()
    assert len(gra.symbols) == 16
    assert (gra.charge, gra.spin) == (0, 0)
    centre = pm.stereo_at(gra, 1)
    bond = pm.stereo_at(gra, (3, 4))
    assert isinstance(centre, pm.Tetrahedral)
    assert isinstance(bond, pm.PlanarBond)
    # the rest are the trivial LS that from_smiles fills (the two CH3 carbons)
    others = [ls for ls in gra.atom_stereo if ls != centre]
    assert sorted(ls.atoms[0] for ls in others) == [0, 5]
    assert gra.bond_stereo == (bond,)


def test_abstraction_ts() -> None:
    """C2H6 + OH: one breaking C-H and one forming O-H, sharing the H."""
    ts = examples.abstraction_ts()
    assert len(ts.symbols) == 10
    assert ts.spin == 1
    ((c, h1),) = ts.breaking
    ((o, h2),) = ts.forming
    assert h1 == h2
    assert ts.symbols[h1] == "H"
    assert (ts.symbols[c], ts.symbols[o]) == ("C", "O")


def test_stereoisomerization_ts() -> None:
    """(Z) -> (E) but-2-en-1-yl: only the PlanarBond at C1=C2 changes."""
    ts = examples.stereoisomerization_ts()
    assert len(ts.symbols) == 11
    assert ts.spin == 1
    assert (ts.breaking, ts.forming) == ((), ())
    (change,) = ts.stereo_changes
    assert change == pm.stereo_at(ts, (1, 2))
    assert isinstance(change.reactant, pm.PlanarBond)
    assert change.product == pm.invert(change.reactant)
    assert isinstance(change.ts, pm.AtropBond)
    assert change.ts.parity == 1
