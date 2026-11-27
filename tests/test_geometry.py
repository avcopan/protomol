"""Tests for from_geometry and embed, between graphs and autogeom geometries.

The geometries come from autogeom's RDKit embedding of a SMILES, which keeps the atom
order `from_smiles` uses (heavy atoms as written, then the H in the order of their
atoms), so the two graphs are index-aligned.
"""

import autogeom as ag
import numpy as np
import pytest

import protomol as pm
from protomol._geometry import _ls_from_coords

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
    # Onium octets (autogeom 0.0.5): a + on C would lose the C=O+ / C=N+ PlanarBond
    r"C/C=[O+]/[O-]",  # syn Criegee intermediate
    r"C/C=[O+]\[O-]",  # anti
    # Not acrolein oxide: SMILES can't write its C-C (s-cis/s-trans, an LS since P54
    # R6); test_low_t.py round-trips it
    r"C/C=[N+](/[O-])C",  # a nitrone
]


@pytest.mark.parametrize(
    ("smiles", "bond"),
    [("CC=[O+][O-]", (1, 2)), ("C=CC=[O+][O-]", (2, 3)), ("CC=[N+]([O-])C", (1, 2))],
)
def test_onium_planar_bond(smiles: str, bond: tuple[int, int]) -> None:
    """The C=O+ or C=N+ bond of an onium zwitterion is a determined PlanarBond."""
    gra = pm.from_geometry(ag.from_smiles(smiles))
    ls = pm.stereo_at(gra, bond)
    assert isinstance(ls, pm.PlanarBond)
    assert ls.parity is not None


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


# embed


def _example_species() -> dict[str, pm.StereoMolGraph]:
    """The species of the examples: the species and each TS's reactants and products."""
    abstraction = pm.examples.abstraction_ts()
    rotation = pm.examples.stereoisomerization_ts()
    return {
        "species": pm.examples.species(),
        "abstraction_reactants": pm.reactants(abstraction),
        "abstraction_products": pm.products(abstraction),
        "rotation_reactants": pm.reactants(rotation),
        "rotation_products": pm.products(rotation),
    }


@pytest.mark.parametrize(("name", "gra"), _example_species().items())
def test_embed_round_trip(name: str, gra: pm.StereoMolGraph) -> None:
    """from_geometry(embed(g)) is g, atom for atom (so the canonical JSON matches)."""
    geo = pm.embed(gra)
    assert geo.symbols == list(gra.symbols), name
    assert (geo.charge, geo.spin) == (gra.charge, gra.spin)
    back = pm.from_geometry(geo)
    assert back == gra
    assert pm.canonical_json(back) == pm.canonical_json(gra)


@pytest.mark.parametrize(
    "smiles",
    [
        *DETERMINED,
        "CN(CC)CCC",  # a lone-pair centre
        "CN(CC)CN(CC)CN(CC)C",  # three of them
        "CC=C=CC",  # an allene axis, which ETKDG doesn't hold
        "C=CC=CC=C",
        "CC(C)(C)C",  # four trivial CH3 centres
    ],
)
def test_embed_from_geometry_graphs(smiles: str) -> None:
    """A determined graph read from a geometry embeds back to itself."""
    gra = pm.from_geometry(ag.from_smiles(smiles))
    assert pm.is_determined(gra)
    assert pm.from_geometry(pm.embed(gra)) == gra


def test_embed_each_configuration() -> None:
    """Both enantiomers, and cis and trans, embed as given."""
    for smiles in (r"C[C@@H](O)/C=C/C", r"C[C@H](O)/C=C\C", r"C[C@@H](O)/C=C\C"):
        gra = pm.from_smiles(smiles)
        assert pm.from_geometry(pm.embed(gra)) == gra


def test_embed_fragments_apart() -> None:
    """Fragments are embedded apart, so no bond is perceived between them."""
    gra = pm.from_smiles("CC.[OH]")
    assert pm.from_geometry(pm.embed(gra)) == gra


def test_embed_reproducible() -> None:
    """The seeds are fixed, so two calls give the same coordinates."""
    gra = pm.examples.species()
    np.testing.assert_array_equal(pm.embed(gra).coordinates, pm.embed(gra).coordinates)


def test_embed_undetermined_and_plain() -> None:
    """Undetermined LS and plain graphs embed; the bonds come back."""
    undetermined = pm.from_smiles("CC(O)C=CC")
    assert not pm.is_determined(undetermined)
    assert pm.from_geometry(pm.embed(undetermined)).bonds == undetermined.bonds
    plain = pm.without_stereo(pm.examples.species())
    assert pm.from_geometry(pm.embed(plain)).bonds == plain.bonds


def test_embed_reaction_graph() -> None:
    """A reaction graph has no single geometry."""
    with pytest.raises(TypeError, match="MolGraph or a StereoMolGraph"):
        pm.embed(pm.examples.abstraction_ts())  # ty: ignore[invalid-argument-type]


def test_embed_no_lewis_structure() -> None:
    """A bare singlet C atom has no Lewis structure."""
    with pytest.raises(pm.GraphError, match="No Lewis structure"):
        pm.embed(pm.MolGraph(symbols=["C"], bonds=[]))


AGREE = pm.GeometryMismatches(
    missing_bonds=(), extra_bonds=(), reacting_bonds=(), stereo=()
)


@pytest.mark.parametrize("smiles", DETERMINED)
def test_geometry_mismatches_agree(smiles: str) -> None:
    """A graph agrees with its own embedding and with a geometry it was read from."""
    gra = pm.from_smiles(smiles)
    assert pm.geometry_mismatches(gra, pm.embed(gra)) == AGREE
    geo = ag.from_smiles(smiles)
    assert pm.geometry_mismatches(pm.from_geometry(geo), geo) == AGREE


def test_geometry_mismatches_stereo() -> None:
    """Each contradicted LS is named by its site, in site order."""
    gra = pm.examples.species()
    epimer = pm.embed(pm.from_smiles(r"C[C@H](O)/C=C/C"))
    assert pm.geometry_mismatches(gra, epimer).stereo == (1,)
    z_isomer = pm.embed(pm.from_smiles(r"C[C@@H](O)/C=C\C"))
    assert pm.geometry_mismatches(gra, z_isomer).stereo == ((3, 4),)
    both = pm.embed(pm.from_smiles(r"C[C@H](O)/C=C\C"))
    assert pm.geometry_mismatches(gra, both).stereo == (1, (3, 4))


def test_geometry_mismatches_undetermined() -> None:
    """An undetermined LS, or a plain graph, has no stereo to contradict."""
    geo = pm.embed(pm.from_smiles(r"C[C@H](O)/C=C\C"))
    assert pm.geometry_mismatches(pm.from_smiles("CC(O)C=CC"), geo) == AGREE
    plain = pm.without_stereo(pm.examples.species())
    assert pm.geometry_mismatches(plain, geo) == AGREE


def test_geometry_mismatches_bool() -> None:
    """True when any field is non-empty (P77)."""
    assert not AGREE
    assert pm.GeometryMismatches(
        missing_bonds=((0, 1),), extra_bonds=(), reacting_bonds=(), stereo=()
    )
    assert pm.GeometryMismatches(
        missing_bonds=(), extra_bonds=((0, 1),), reacting_bonds=(), stereo=()
    )
    assert pm.GeometryMismatches(
        missing_bonds=(), extra_bonds=(), reacting_bonds=(), stereo=(1,)
    )
    assert pm.GeometryMismatches(
        missing_bonds=(), extra_bonds=(), reacting_bonds=((0, 1),), stereo=()
    )


def test_geometry_mismatches_bonds() -> None:
    """A wrong bond is missing (in the graph only) and extra (in the geometry only)."""
    gra = pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (1, 2)])
    geo = pm.embed(pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)]))
    result = pm.geometry_mismatches(gra, geo)
    assert result.missing_bonds == ((1, 2),)
    assert result.extra_bonds == ((0, 2),)
    assert result.stereo == ()


def test_geometry_mismatches_end_points() -> None:
    """A reaction graph's geometry is its TS's: an end point's bonds aren't partial."""
    ts = pm.examples.abstraction_ts()  # C0-H3 breaks, O2-H3 forms
    rct, prd = pm.reactants(ts), pm.products(ts)
    assert pm.geometry_mismatches(rct, pm.embed(rct)) == AGREE
    for side in (rct, prd):  # one bond full, the other far
        result = pm.geometry_mismatches(ts, pm.embed(side))
        assert result.reacting_bonds == ((0, 3), (2, 3))
    # A TS: H3 between C0 and O2, both bonds partial
    geo = pm.embed(rct)
    xyz = np.array(geo.coordinates)
    axis = (xyz[3] - xyz[0]) / np.linalg.norm(xyz[3] - xyz[0])
    xyz[3] = xyz[0] + 1.3 * axis
    xyz[2] = xyz[3] + 1.25 * axis
    side = np.cross(axis, [0.0, 0.0, 1.0])
    xyz[9] = xyz[2] + 0.97 * side / np.linalg.norm(side)
    ts_geo = ag.Geometry(symbols=geo.symbols, coordinates=xyz, charge=0, spin=1)
    assert pm.geometry_mismatches(ts, ts_geo) == AGREE


def _sn2_ts_geometry(srg: pm.StereoReactionGraph, center: int) -> ag.Geometry:
    """A backside SN2 TS from an embedding of the reactants.

    The leaving atom stays on its axis at 1.3x its bond, the other neighbours fold
    into the plane through the centre, and the attacking atom comes in from the other
    side.
    """
    ((leaving,),) = [set(b) - {center} for b in srg.breaking]
    ((attacking,),) = [set(b) - {center} for b in srg.forming]
    geo = pm.embed(pm.reactants(srg))
    xyz = np.array(geo.coordinates)
    axis = xyz[leaving] - xyz[center]
    length = np.linalg.norm(axis)
    axis /= length
    for nbr in pm.neighbors(pm.reactants(srg), center):
        if nbr != leaving:
            vec = xyz[nbr] - xyz[center]
            flat = vec - np.dot(vec, axis) * axis
            xyz[nbr] = xyz[center] + flat / np.linalg.norm(flat) * np.linalg.norm(vec)
    xyz[leaving] = xyz[center] + 1.3 * length * axis
    xyz[attacking] = xyz[center] - 1.3 * 2.14 * axis  # C-I: 2.14 A
    return ag.Geometry(
        symbols=geo.symbols, coordinates=xyz, charge=geo.charge, spin=geo.spin
    )


def test_geometry_mismatches_sn2_ts() -> None:
    """An SN2 TS geometry reads as the backside TBP; its mirror image doesn't."""
    rct, prd = pm.from_smiles("F[C@H](Cl)Br.[I-]"), pm.from_smiles("F[C@@H](Cl)I.[Br-]")
    (backside,) = pm.ts_graphs(rct, prd)
    geo = _sn2_ts_geometry(backside, 1)
    (change,) = backside.stereo_changes
    assert isinstance(change.ts, pm.TrigonalBipyramidal)
    xyz = np.asarray(geo.coordinates)
    assert _ls_from_coords(change.ts, xyz) == change.ts  # read, not just undecided
    assert pm.geometry_mismatches(backside, geo) == AGREE
    mirror = ag.reflect(geo, [0.0, 0.0, 1.0])
    assert pm.geometry_mismatches(backside, mirror).stereo == (1,)


def test_geometry_mismatches_twisted_ts() -> None:
    """A twisted but-2-ene TS agrees with one AtropBond parity; reflected, the other."""
    plus, minus = pm.ts_graphs(pm.from_smiles(r"C/C=C\C"), pm.from_smiles("C/C=C/C"))
    geo = ag.twist(pm.embed(pm.reactants(plus)), (1, 2), 90, degrees=True)
    mirror = ag.reflect(geo, [0.0, 0.0, 1.0])
    # The reflection also inverts the CH3 sites' trivial LS: look at C1=C2 only
    agree = [
        [(1, 2) not in pm.geometry_mismatches(ts, g).stereo for ts in (plus, minus)]
        for g in (geo, mirror)
    ]
    assert agree in ([[True, False], [False, True]], [[False, True], [True, False]])
    # An end point is planar: neither twist is contradicted
    assert pm.geometry_mismatches(plus, pm.embed(pm.reactants(plus))) == AGREE


def test_geometry_mismatches_reaction_stereo() -> None:
    """The unchanged LS and the ts slots are compared on a reaction graph."""
    ts = pm.examples.stereoisomerization_ts()
    for side in (pm.reactants(ts), pm.products(ts)):
        assert pm.geometry_mismatches(ts, pm.embed(side)) == AGREE
    # Each side's geometry contradicts the other side at the changing bond
    rct, prd = pm.reactants(ts), pm.products(ts)
    assert pm.geometry_mismatches(rct, pm.embed(prd)).stereo == ((1, 2),)


def test_geometry_mismatches_wrong_atoms() -> None:
    """The geometry must have the graph's atoms, in order."""
    geo = pm.embed(pm.from_smiles("CO"))
    with pytest.raises(pm.GraphError, match="differ"):
        pm.geometry_mismatches(pm.from_smiles("OC"), geo)
