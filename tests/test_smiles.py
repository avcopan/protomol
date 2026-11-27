"""Tests for from_smiles."""

import warnings

import pytest

import protomol as pm
from protomol._backend import from_smg, to_smg
from protomol._smiles import _without_linear_bond_stereo


def _undetermined_sites(g: pm.StereoMolGraph) -> list[int | tuple[int, int]]:
    return [pm.site(ls) for ls in (*g.atom_stereo, *g.bond_stereo) if ls.parity is None]


@pytest.mark.parametrize(
    ("smiles", "charge", "spin"),
    [
        ("C", 0, 0),
        ("[CH3]", 0, 1),
        ("[CH2]", 0, 2),
        ("[O]", 0, 2),
        ("O=O", 0, 0),
        ("[OH]", 0, 1),
        ("[NH4+]", 1, 0),
        ("C=C[CH2]", 0, 1),
    ],
)
def test_default_charge_and_spin(smiles: str, charge: int, spin: int) -> None:
    """Charge is the formal charges' sum, spin RDKit's radical electrons."""
    g = pm.from_smiles(smiles)
    assert (g.charge, g.spin) == (charge, spin)


def test_explicit_charge_and_spin() -> None:
    """An explicit charge or spin is used."""
    assert pm.from_smiles("O=O", spin=2).spin == 2
    assert pm.from_smiles("[NH4+]", charge=1).charge == 1


def test_wrong_spin_parity_raises() -> None:
    """A spin of the wrong parity raises GraphError."""
    with pytest.raises(pm.GraphError, match="parity"):
        pm.from_smiles("[CH3]", spin=0)


def test_charge_mismatch_raises() -> None:
    """A charge that differs from the formal charges raises GraphError."""
    with pytest.raises(pm.GraphError, match="formal charges"):
        pm.from_smiles("[NH4+]", charge=0)


def test_from_smiles_rejects_reaction_smiles() -> None:
    """A reaction SMILES raises GraphError."""
    with pytest.raises(pm.GraphError, match="reaction SMILES"):
        pm.from_smiles("CC.[OH]>>C[CH2].O")


def test_unparsable_smiles_raises() -> None:
    """A SMILES RDKit can't parse raises GraphError."""
    with pytest.raises(pm.GraphError, match="parse"):
        pm.from_smiles("C(C")


def test_atoms_in_smiles_order() -> None:
    """Heavy atoms in SMILES order, then implicit H by parent atom."""
    g = pm.from_smiles("OCN")
    assert g.symbols == ("O", "C", "N", "H", "H", "H", "H", "H")
    parents = [pm.neighbors(g, h)[0] for h in range(3, 8)]
    assert parents == [0, 1, 1, 2, 2]


def test_bracket_h_keeps_its_position() -> None:
    """A [H] written in the SMILES keeps its place (stereomol's index rule)."""
    g = pm.from_smiles("[H]OC")
    assert g.symbols == ("H", "O", "C", "H", "H", "H")
    assert pm.neighbors(g, 0) == (1,)
    g = pm.from_smiles("C[C@@]([H])(O)F")
    assert g.symbols[2] == "H"
    assert pm.neighbors(g, 2) == (1,)


def test_fragments() -> None:
    """A SMILES of several fragments gives one graph."""
    g = pm.from_smiles("CC.[OH]")
    assert g.symbols[:3] == ("C", "C", "O")
    assert (g.charge, g.spin) == (0, 1)


def test_methane_is_determined() -> None:
    """CH4's C is trivial, so it is filled."""
    g = pm.from_smiles("C")
    assert len(g.atom_stereo) == 1
    assert pm.is_determined(g)


@pytest.mark.parametrize("smiles", ["CC", "C=C", "N", "CC=O", "C1CC1", "c1ccccc1"])
def test_trivial_sites_are_filled(smiles: str) -> None:
    """Sites whose configurations give one graph are filled."""
    assert pm.is_determined(pm.from_smiles(smiles))


def test_unspecified_stereocentre_stays_undetermined() -> None:
    """Butan-2-ol's C1 stays undetermined."""
    assert _undetermined_sites(pm.from_smiles("CC(O)CC")) == [1]


def test_unspecified_double_bond_stays_undetermined() -> None:
    """But-2-ene's double bond stays undetermined."""
    assert _undetermined_sites(pm.from_smiles("CC=CC")) == [(1, 2)]


def test_meso_centres_stay_undetermined() -> None:
    """Each centre of 2,3-butanediol is stereogenic on its own."""
    assert _undetermined_sites(pm.from_smiles("CC(O)C(C)O")) == [1, 3]


def test_specified_stereo_is_kept() -> None:
    """The SMILES's centre and double-bond stereo are determined."""
    g = pm.from_smiles("C[C@@H](O)/C=C/C")
    assert pm.is_determined(g)
    centre, bond = pm.stereo_at(g, 1), pm.stereo_at(g, (3, 4))
    assert isinstance(centre, pm.Tetrahedral)
    assert centre.parity in (1, -1)
    assert isinstance(bond, pm.PlanarBond)
    assert bond.parity == 0


def test_specified_stereo_distinguishes_isomers() -> None:
    """Opposite configurations give different graphs."""
    assert pm.from_smiles("C[C@@H](O)F") != pm.from_smiles("C[C@H](O)F")
    assert pm.from_smiles("C/C=C/C") != pm.from_smiles("C/C=C\\C")


def test_allyl_radical_has_resonance_planar_bonds() -> None:
    """Both C-C bonds of the allyl radical get a PlanarBond (resonance)."""
    g = pm.from_smiles("C=C[CH2]")
    assert {pm.site(ls) for ls in g.bond_stereo} == {(0, 1), (1, 2)}
    assert all(isinstance(ls, pm.PlanarBond) for ls in g.bond_stereo)


def test_amine_n_has_lone_pair_stereo() -> None:
    """N-methyl-N-ethylpropan-1-amine: an undetermined N; NH3's N is trivial."""
    g = pm.from_smiles("CN(CC)CCC")
    ls = pm.stereo_at(g, 1)
    assert isinstance(ls, pm.Tetrahedral)
    assert None in ls.atoms
    assert ls.parity is None
    ls = pm.stereo_at(pm.from_smiles("N"), 0)
    assert isinstance(ls, pm.Tetrahedral)
    assert None in ls.atoms
    assert ls.parity is not None


def test_symmetric_amine_n_is_trivial() -> None:
    """N,N-dimethylethanamine: swapping the methyls mirrors the N, so it's filled."""
    ls = pm.stereo_at(pm.from_smiles("CN(C)CC"), 1)
    assert isinstance(ls, pm.Tetrahedral)
    assert ls.parity is not None


@pytest.mark.parametrize("smiles", ["CC(=O)NC", "Nc1ccccc1"])
def test_amide_and_aniline_n_have_no_lone_pair_stereo(smiles: str) -> None:
    """Amide and aniline N are conjugated, so they get no LS (more in test_low_t)."""
    n = pm.from_smiles(smiles).symbols.index("N")
    assert pm.stereo_at(pm.from_smiles(smiles), n) is None


# The back end (private)


def _from_smiles_quietly(smiles: str) -> pm.StereoMolGraph:
    """from_smiles, failing on any warning (no chain LS may reach stereomolgraph)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return pm.from_smiles(smiles)


def test_allene_is_trivial() -> None:
    """C=C=C: one determined AtropBond on (0, 2) with chain (1,)."""
    g = _from_smiles_quietly("C=C=C")
    assert len(g.bond_stereo) == 1
    ls = pm.stereo_at(g, (0, 2))
    assert isinstance(ls, pm.AtropBond)
    assert ls.chain == (1,)
    assert ls.parity is not None


def test_dimethylallene_is_undetermined() -> None:
    """CC=C=CC: an undetermined AtropBond on (1, 3) with chain (2,)."""
    g = _from_smiles_quietly("CC=C=CC")
    assert len(g.bond_stereo) == 1
    ls = pm.stereo_at(g, (1, 3))
    assert isinstance(ls, pm.AtropBond)
    assert ls.chain == (2,)
    assert ls.parity is None


def test_butatriene_gets_planar_chain() -> None:
    """CC=C=C=CC: an undetermined PlanarBond on (1, 4) with chain (2, 3)."""
    g = _from_smiles_quietly("CC=C=C=CC")
    assert len(g.bond_stereo) == 1
    ls = pm.stereo_at(g, (1, 4))
    assert isinstance(ls, pm.PlanarBond)
    assert ls.chain == (2, 3)
    assert ls.parity is None


def test_pentatetraene_gets_atrop_chain() -> None:
    """CC=C=C=C=CC: four double bonds, an AtropBond on (1, 5) with chain (2, 3, 4)."""
    g = _from_smiles_quietly("CC=C=C=C=CC")
    ls = pm.stereo_at(g, (1, 5))
    assert isinstance(ls, pm.AtropBond)
    assert ls.chain == (2, 3, 4)


@pytest.mark.parametrize("smiles", ["C=C=C", "CC=C=CC", "CC=C=C=CC", "C=C=CC=C=C"])
def test_no_per_bond_ls_inside_chain(smiles: str) -> None:
    """Every bond LS is a chain LS, none on a chain's double bonds."""
    g = _from_smiles_quietly(smiles)
    assert g.bond_stereo
    assert all(ls.chain for ls in g.bond_stereo)


def test_two_chains_in_one_graph() -> None:
    """C=C=CC=C=C: two trivial allene axes, (0, 2) and (3, 5)."""
    g = _from_smiles_quietly("C=C=CC=C=C")
    assert [pm.site(ls) for ls in g.bond_stereo] == [(0, 2), (3, 5)]
    assert pm.is_determined(g)


def test_chain_in_a_ring() -> None:
    """Cyclonona-1,2-diene: each end's substituents differ, so it's undetermined."""
    g = _from_smiles_quietly("C1CCCCCC=C=C1")
    assert _undetermined_sites(g) == [(6, 8)]


@pytest.mark.parametrize("smiles", ["CC=C=O", "C=[N+]=[N-]", "O=C=O"])
def test_chain_with_a_bare_end_has_no_ls(smiles: str) -> None:
    """A ketene, diazomethane and CO2: an end without two substituents, so no LS."""
    assert not _from_smiles_quietly(smiles).bond_stereo


def test_ketenimine_axis_has_a_lone_pair_slot() -> None:
    """CC=C=NC: the bent N end has one substituent and a lone pair, as in an imine."""
    g = _from_smiles_quietly("CC=C=NC")
    ls = pm.stereo_at(g, (1, 3))
    assert isinstance(ls, pm.AtropBond)
    assert ls.chain == (2,)
    assert None in ls.atoms
    assert ls.parity is None


def test_bond_stereo_needs_sp2_ends() -> None:
    """A PlanarBond with a linear (sp) end, the C of a ketene, is removed."""
    ketene = pm.StereoMolGraph(  # CH3-CH=C=O: C0, C1, C2, O3, H4-H7
        symbols=["C", "C", "C", "O", "H", "H", "H", "H"],
        bonds=[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (0, 6), (1, 7)],
        bond_stereo=[pm.PlanarBond(atoms=(0, 7, 1, 2, 3, None))],
    )
    assert not _without_linear_bond_stereo(ketene, linear={2}).bond_stereo
    assert _without_linear_bond_stereo(ketene, linear=set()).bond_stereo


def _sn2() -> pm.StereoReactionGraph:
    tet = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
    change = pm.StereoChange(
        reactant=tet,
        product=pm.Tetrahedral(atoms=(0, 1, 2, 5, 4), parity=1),
        ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
    )
    return pm.StereoReactionGraph(
        symbols=["C", "F", "Cl", "Br", "H", "I"],
        bonds=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        breaking=[(0, 3)],
        forming=[(0, 5)],
        charge=-1,
        stereo_changes=[change],
    )


@pytest.mark.parametrize(
    "g",
    [
        pm.MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)]),
        pm.from_smiles("C[C@@H](O)/C=C/C"),
        pm.ReactionGraph(
            symbols=["H", "H", "H"],
            bonds=[(0, 1), (1, 2)],
            breaking=[(0, 1)],
            forming=[(1, 2)],
        ),
        _sn2(),
    ],
    ids=["MolGraph", "StereoMolGraph", "ReactionGraph", "StereoReactionGraph"],
)
def test_backend_round_trip(g: pm.AnyGraph) -> None:
    """from_smg(to_smg(g)) == g for each type."""
    assert from_smg(to_smg(g), charge=g.charge, spin=g.spin) == g


def test_backend_drops_chain_ls_with_a_warning() -> None:
    """A chain LS is dropped with a warning; other graphs don't warn."""
    allene = pm.StereoMolGraph(  # H2C=C=CH2: C0, C1, C2, H3-H6
        symbols=["C", "C", "C", "H", "H", "H", "H"],
        bonds=[(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)],
        bond_stereo=[pm.AtropBond(atoms=(3, 4, 0, 2, 5, 6), chain=(1,))],
    )
    with pytest.warns(UserWarning, match="chain"):
        smg = to_smg(allene)
    assert not smg.bond_stereo
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        to_smg(pm.from_smiles("C=C"))


def test_reaction_smiles_atoms_follow_map_numbers() -> None:
    """Atom k is the atom with the k-th smallest map number, on both sides."""
    g = pm.from_reaction_smiles("[H:5][O:3][H:1].[H:4]>>[H:5][O:3].[H:1][H:4]")
    assert g.symbols == ("H", "O", "H", "H")
    assert (g.breaking, g.forming) == (((0, 1),), ((0, 2),))


def test_reaction_smiles_stereo_independent_of_smiles_order() -> None:
    """Two writings of the same mapped reaction give the same graph (LS relabelled)."""
    prd = ">>[C:1]([F:2])([Cl:3])[Br:4].[H:5][O:6]"
    rxn1 = "[C@:1]([F:2])([Cl:3])([Br:4])[H:5].[O:6]" + prd
    rxn2 = "[Cl:3][C@@:1]([F:2])([Br:4])[H:5].[O:6]" + prd.replace(
        "[H:5][O:6]", "[O:6][H:5]"
    )
    g1, g2 = pm.from_reaction_smiles(rxn1), pm.from_reaction_smiles(rxn2)
    assert g1 == g2
    assert pm.reactants(g1) == pm.from_smiles("[C@](F)(Cl)(Br)[H].[O]")
    (change,) = g1.stereo_changes
    assert change.reactant is not None
    assert change.reactant.parity is not None
    assert change.ts == change.reactant  # still pyramidal at the TS (P70)


def test_reaction_smiles_spin_default_is_the_lower_side() -> None:
    """The default spin is the lower of the sides' radical counts; it can be given."""
    rxn = "[C:1]([H:2])([H:3])=[C:4]([H:5])[H:6]>>[C:1]([H:2])([H:3])[C:4]([H:5])[H:6]"
    assert pm.from_reaction_smiles(rxn).spin == 0
    assert pm.from_reaction_smiles(rxn, spin=2).spin == 2


def test_reaction_smiles_charge() -> None:
    """The charge is the formal charges' sum, and an explicit one is checked."""
    rxn = "[N+:1]([H:2])([H:3])([H:4])[H:5]>>[N:1]([H:2])([H:3])[H:4].[H+:5]"
    assert pm.from_reaction_smiles(rxn).charge == 1
    with pytest.raises(pm.GraphError, match="formal charges"):
        pm.from_reaction_smiles(rxn, charge=0)


@pytest.mark.parametrize(
    ("smiles", "match"),
    [
        ("[H:1][H:2]", "not a reaction SMILES"),
        ("[H:1][H:2]>[O:3]>[H:1].[H:2]", "agents"),
        ("[H:1][H:2]>>[H:1].[H]", "unmapped"),
        ("[CH4:1]>>[CH3:1].[H:2]", "implicit H"),
        ("[H:1][H:1]>>[H:1].[H:1]", "twice"),
        ("[H:1][H:2]>>[H:1].[H:3]", "different map numbers"),
        ("[H:1][H:2]>>[H:1].[F:2]", "symbols"),
        ("[H:1]([H:2]>>[H:1].[H:2]", "parse"),
    ],
)
def test_bad_reaction_smiles_raises(smiles: str, match: str) -> None:
    """A malformed or partly mapped reaction SMILES raises GraphError."""
    with pytest.raises(pm.GraphError, match=match):
        pm.from_reaction_smiles(smiles)
