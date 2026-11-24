"""Tests for from_geometry: bonds, LS sites and parities read from an autogeom geometry.

The geometries come from autogeom's RDKit embedding of a SMILES, which keeps the atom
order `from_smiles` uses (heavy atoms as written, then the H in the order of their
atoms), so the two graphs are index-aligned.
"""

import autogeom as ag
import numpy as np
import pytest

import protomol as pm

SPECIES_SMILES = r"C[C@@H](O)/C=C/C"  # pm.examples.species()

# SMILES whose every stereogenic site is given, so the canonical graphs must match
DETERMINED = [
    SPECIES_SMILES,
    r"C[C@H](O)/C=C\C",
    r"[CH2]/C=C\C",
    "O[C@H]1CCCC[C@@H]1O",
    "O[C@H]1CCCC[C@H]1O",
    "C[C@@H](O)[C@H](C)O",
    "CC(=O)[O-]",
    "C[N+](=O)[O-]",
    "C=C[CH2]",
    "[O][O]",
]


def _permuted(geo: ag.Geometry, perm: np.ndarray) -> ag.Geometry:
    """The geometry whose atom k is atom perm[k] of `geo`."""
    return ag.Geometry(
        symbols=[geo.symbols[i] for i in perm],
        coordinates=geo.coordinates[perm],
        charge=geo.charge,
        spin=geo.spin,
    )


@pytest.mark.parametrize("smiles", DETERMINED)
def test_matches_from_smiles(smiles: str) -> None:
    """Same atoms, bonds and canonical graph as `from_smiles`, and fully determined."""
    geo = ag.from_smiles(smiles)
    gra = pm.from_geometry(geo)
    ref = pm.from_smiles(smiles, spin=geo.spin)
    assert gra.symbols == ref.symbols
    assert gra.bonds == ref.bonds
    assert (gra.charge, gra.spin) == (geo.charge, geo.spin)
    assert pm.is_determined(gra)
    assert pm.canonical_json(gra) == pm.canonical_json(ref)


def test_species_example() -> None:
    """The species example, from a geometry."""
    gra = pm.from_geometry(ag.from_smiles(SPECIES_SMILES))
    assert pm.canonical_json(gra) == pm.canonical_json(pm.examples.species())


@pytest.mark.parametrize("smiles", [SPECIES_SMILES, "CN(CC)CCC", "C=C[CH2]"])
def test_atom_order_kept(smiles: str) -> None:
    """Renumbering the geometry renumbers the graph the same way, LS included."""
    geo = ag.from_smiles(smiles)
    gra = pm.from_geometry(geo)
    rng = np.random.default_rng(0)
    for _ in range(10):
        perm = rng.permutation(len(geo.symbols))
        atom_map = tuple(int(i) for i in perm)
        assert pm.from_geometry(_permuted(geo, perm)) == pm.relabel(gra, atom_map)


@pytest.mark.parametrize(
    ("smiles", "chiral"),
    [
        (SPECIES_SMILES, True),
        ("O[C@H]1CCCC[C@@H]1O", True),  # trans: (R,R)
        ("O[C@H]1CCCC[C@H]1O", False),  # cis: meso
        ("C[C@@H](O)[C@H](C)O", False),  # meso butane-2,3-diol
        ("CN(CC)CCC", True),  # a pyramidal amine N
    ],
)
def test_mirror_image(smiles: str, *, chiral: bool) -> None:
    """A reflected geometry gives the enantiomer: a different graph only if chiral."""
    geo = ag.from_smiles(smiles)
    gra = pm.from_geometry(geo)
    mirror = pm.from_geometry(ag.reflect(geo, [0.0, 0.0, 1.0]))
    assert (pm.canonical_json(mirror) != pm.canonical_json(gra)) == chiral


def test_enantiomer_from_smiles() -> None:
    """The reflected species is its SMILES enantiomer."""
    mirror = pm.from_geometry(ag.reflect(ag.from_smiles(SPECIES_SMILES), [1.0, 0, 0]))
    expected = pm.from_smiles(r"C[C@H](O)/C=C/C")
    assert pm.canonical_json(mirror) == pm.canonical_json(expected)


def test_amine_determined() -> None:
    """An amine N gets its lone-pair LS from the coordinates."""
    gra = pm.from_geometry(ag.from_smiles("CN(CC)CCC"))
    (nitrogen,) = (ls for ls in gra.atom_stereo if gra.symbols[ls.atoms[0]] == "N")
    assert nitrogen.parity is not None


def test_flat_amine_undetermined() -> None:
    """A planar NH3 can't be read, so its LS stays undetermined."""
    geo = ag.Geometry(
        symbols=["N", "H", "H", "H"],
        coordinates=[[0, 0, 0], [1.0, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]],
        charge=0,
        spin=0,
    )
    (ls,) = pm.from_geometry(geo).atom_stereo
    assert ls.parity is None


def test_twisted_double_bond_undetermined() -> None:
    """Ethylene twisted by 90° has no cis or trans."""
    geo = ag.from_smiles("C=C")
    xyz = geo.coordinates.copy()
    center = xyz[1]
    axis = (xyz[1] - xyz[0]) / np.linalg.norm(xyz[1] - xyz[0])
    for i in (4, 5):  # the H on C1: turn them 90° about the C=C axis
        v = xyz[i] - center
        par = np.dot(v, axis) * axis
        perp = v - par
        xyz[i] = center + par + np.cross(axis, perp)
    twisted = ag.Geometry(symbols=geo.symbols, coordinates=xyz, charge=0, spin=0)
    (ls,) = pm.from_geometry(twisted).bond_stereo
    assert ls.parity is None


def test_no_lewis_structure() -> None:
    """A bare C atom can't be a closed-shell singlet with no bonds."""
    geo = ag.Geometry(symbols=["C"], coordinates=[[0, 0, 0]], charge=0, spin=0)
    with pytest.raises(pm.GraphError, match="No Lewis structure"):
        pm.from_geometry(geo)
