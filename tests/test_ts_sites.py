"""Tests for TS sites (P67-P74): the reacting core, the `ts` slots, the test reactions.

The reactions and their expected TS graphs are the tables of the TS-sites guide
(protomol's planning repo, `guides/2026-11-25-ts-sites.md`); atom indices are those of
`from_smiles` on the reactant side.
"""

import pytest

import protomol as pm
from protomol._ts import _reacting_core, _ts_sites


def _channels(rct: str, prd: str) -> list[pm.StereoReactionGraph]:
    return pm.ts_graphs(pm.from_smiles(rct), pm.from_smiles(prd))


@pytest.mark.parametrize(
    ("rct", "prd", "core", "sites"),
    [
        # 1: SN2, C0 attacked by I2 with Br1 leaving
        ("CBr.[I-]", "CI.[Br-]", {0, 1, 2}, (0,)),
        # 3: rotation about C1=C2, no bond change: the bond's ends. The allyl
        # C0-C1 bond's trivial PlanarBond (unchanged) has an end in the core too
        (r"[CH2]/C=C\C", r"[CH2]/C=C/C", {1, 2}, ((0, 1), (1, 2))),
        # 5: H8 abstracted from C1 by O4
        ("CCCC.[OH]", "C[CH]CC.O", {1, 4, 8}, (1,)),
        # 8: OH adds to C1 of the C1=C2 stereo bond: the centre and the bond
        ("C/C=C/C.[OH]", "CC(O)[CH]C", {1, 4}, (1, (1, 2))),
    ],
)
def test_reacting_core(rct: str, prd: str, core: set[int], sites: tuple) -> None:
    """The core is the atoms of the changing bonds, else the changing site's atoms."""
    for srg in _channels(rct, prd):
        assert _reacting_core(srg) == core
        plain = core if srg.breaking or srg.forming else set()
        assert _reacting_core(pm.without_stereo(srg)) == plain
        assert _ts_sites(srg) == sites


def _change(
    srg: pm.StereoReactionGraph, site: int | tuple[int, int]
) -> pm.StereoChange:
    item = pm.stereo_at(srg, site)
    assert isinstance(item, pm.StereoChange)
    return item


def _ts(srg: pm.StereoReactionGraph, site: int | tuple[int, int]) -> pm.Stereo:
    """The site's ts slot, which must be an LS."""
    ts = _change(srg, site).ts
    assert ts is not None
    return ts


def _reverse_is_a_channel(srg: pm.StereoReactionGraph) -> bool:
    """The reverse search finds the reverse TS, ts slots included (P81)."""
    back = pm.ts_graphs(pm.products(srg), pm.reactants(srg))
    return any(pm.isomorphism(b, pm.reverse(srg)) is not None for b in back)


def test_sn2_methyl() -> None:
    """1: backside gives the TBP with Br and I axial, frontside the shared position."""
    channels = _channels("CBr.[I-]", "CI.[Br-]")
    slots = sorted((_ts(g, 0) for g in channels), key=lambda ls: ls.kind)
    assert slots == [
        pm.Tetrahedral(atoms=(0, None, 3, 4, 5), parity=1),
        pm.TrigonalBipyramidal(atoms=(0, 1, 2, 3, 4, 5), parity=1),
    ]
    assert all(_reverse_is_a_channel(g) for g in channels)


@pytest.mark.parametrize(
    ("prd", "kind"),
    [
        ("F[C@@H](Cl)I.[Br-]", "TrigonalBipyramidal"),  # inversion: backside only
        ("F[C@H](Cl)I.[Br-]", "Tetrahedral"),  # retention: frontside only
    ],
)
def test_sn2_chiral(prd: str, kind: str) -> None:
    """1: at a chiral centre each product configuration is one channel."""
    (srg,) = _channels("F[C@H](Cl)Br.[I-]", prd)
    ts = _change(srg, 1).ts
    assert ts is not None
    assert ts.kind == kind
    if isinstance(ts, pm.TrigonalBipyramidal):
        assert {ts.atoms[1], ts.atoms[2]} == {3, 4}  # Br and I axial
    else:
        assert None in ts.atoms
    assert _reverse_is_a_channel(srg)


@pytest.mark.parametrize("prd", [r"C/C(O)=[C]/F", r"C/C(O)=[C]\F"])
def test_bifurcation(prd: str) -> None:
    """2: OH + 1-fluoropropyne: one channel per product, the ts slot undetermined."""
    (srg,) = _channels("CC#CF.[OH]", prd)
    change = _change(srg, (1, 2))
    assert change.reactant is None
    assert change.product is not None
    assert change.product.parity == 0
    assert change.ts == pm.PlanarBond(atoms=change.product.atoms)
    assert _reverse_is_a_channel(srg)  # an alkenyl β-scission: undetermined too


def test_radical_rotation() -> None:
    """3: (Z)→(E) but-2-en-1-yl: the two twisted TSs, mirror images (P39)."""
    channels = _channels(r"[CH2]/C=C\C", r"[CH2]/C=C/C")
    slots = [_ts(g, (1, 2)) for g in channels]
    assert all(isinstance(ts, pm.AtropBond) for ts in slots)
    assert {ts.parity for ts in slots} == {-1, 1}
    assert pm.examples.stereoisomerization_ts() in channels


def test_butene_rotation() -> None:
    """4: (Z)→(E) but-2-ene: two TSs, kept apart by AtropBond's helicity (P45)."""
    plus, minus = _channels(r"C/C=C\C", r"C/C=C/C")
    assert {_ts(g, (1, 2)).parity for g in (plus, minus)} == {1, -1}
    assert pm.isomorphism(plus, minus) is None
    assert pm.is_chiral(plus)
    assert pm.is_chiral(minus)


@pytest.mark.parametrize(
    ("rct", "prd", "centre"),
    [
        ("CCCC.[OH]", "C[CH]CC.O", 1),  # the two channels are mirror images
        ("C[C@@H](O)CC.[OH]", "C[C@@H](O)[CH]C.O", 3),  # diastereomers
    ],
)
def test_abstraction(rct: str, prd: str, centre: int) -> None:
    """5: H abstraction at a CH2: the ts slot is the reactant's (still pyramidal)."""
    channels = _channels(rct, prd)
    assert len(channels) == 2
    for srg in channels:
        change = _change(srg, centre)
        assert change.product is None
        assert change.ts == change.reactant
        assert _reverse_is_a_channel(srg)
    mirrored = pm.isomorphism(pm.enantiomer(channels[0]), channels[1]) is not None
    assert mirrored == (centre == 1)


def test_nitrogen_inversion() -> None:
    """6: a lone-pair inversion is planar at the TS: the slot is None by rule."""
    (srg,) = _channels("C[N@](CC)CCC", "C[N@@](CC)CCC")
    change = _change(srg, 1)
    assert change.reactant is not None
    assert change.product is not None
    assert change.ts is None


@pytest.mark.parametrize("prd", ["[CH2][C@H](C)CC", "[CH2][C@@H](C)CC"])
def test_attacked_face(prd: str) -> None:
    """7: CH3 + but-1-ene: the face attacked fixes the new centre at the TS."""
    (srg,) = _channels("C=CCC.[CH3]", prd)
    change = _change(srg, 1)
    assert change.reactant is None
    assert change.ts == change.product
    assert _reverse_is_a_channel(srg)


def test_attacked_faces_are_mirror_images() -> None:
    """7: the two faces give mirror-image channels."""
    (a,) = _channels("C=CCC.[CH3]", "[CH2][C@H](C)CC")
    (b,) = _channels("C=CCC.[CH3]", "[CH2][C@@H](C)CC")
    assert pm.isomorphism(pm.enantiomer(a), b) is not None


@pytest.mark.parametrize("prd", ["C[C@H](O)[CH]C", "C[C@@H](O)[CH]C"])
def test_addition_to_a_stereo_bond(prd: str) -> None:
    """8: OH + (E)-but-2-ene: the C=C stays planar at the TS; C1 is an attacked face."""
    (srg,) = _channels("C/C=C/C.[OH]", prd)
    bond = _change(srg, (1, 2))
    assert bond.product is None
    assert bond.ts == bond.reactant
    centre = _change(srg, 1)
    assert centre.ts == centre.product
    assert _reverse_is_a_channel(srg)  # β-scission gives the same TS (P81)


def test_builders_agree() -> None:
    """reaction_graph and low_t set the slots as ts_graphs does (P73)."""
    (srg,) = _channels("F[C@H](Cl)Br.[I-]", "F[C@@H](Cl)I.[Br-]")
    assert pm.reaction_graph(pm.reactants(srg), pm.products(srg)) == srg
    assert pm.low_t(srg) == srg
    rotation = pm.reaction_graph(
        pm.from_smiles(r"C/C=C\C"), pm.from_smiles("C/C=C/C")
    )  # the rules allow both twists: unchosen
    assert _ts(rotation, (1, 2)) == pm.AtropBond(atoms=_ts(rotation, (1, 2)).atoms)
    for channel in _channels(r"C/C=C\C", "C/C=C/C"):
        assert pm.low_t(channel) == channel  # a chosen twist is kept


def test_constructor_does_not_infer() -> None:
    """A hand-built graph keeps what it is given (P73)."""
    srg = pm.examples.abstraction_ts()
    (change,) = srg.stereo_changes
    bare = pm.with_stereo(srg, pm.StereoChange(reactant=change.reactant))
    assert bare.stereo_changes[0].ts is None
