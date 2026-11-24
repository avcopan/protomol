"""Tests for the local stereodescriptors (LS)."""

import itertools

import pytest
from pydantic import ValidationError

import protomol as pm
from protomol._stereo import _FULL_GROUP, _GROUP, _INVERSION

type LS = (
    pm.Tetrahedral
    | pm.SquarePlanar
    | pm.TrigonalBipyramidal
    | pm.Octahedral
    | pm.PlanarBond
    | pm.AtropBond
)

DETERMINED: list[LS] = [
    pm.Tetrahedral(atoms=(0, 4, 2, None, 1), parity=1),
    pm.Tetrahedral(atoms=(0, 4, 2, 3, 1), parity=-1),
    pm.SquarePlanar(atoms=(0, 4, 2, 3, 1), parity=0),
    pm.TrigonalBipyramidal(atoms=(0, 5, 3, 1, 2, 4), parity=1),
    pm.TrigonalBipyramidal(atoms=(0, 5, None, 1, 2, 4), parity=-1),
    pm.Octahedral(atoms=(0, 6, 2, 5, 3, 1, 4), parity=1),
    pm.PlanarBond(atoms=(8, None, 3, 1, 9, 5), parity=0),
    pm.PlanarBond(atoms=(8, 7, 3, 1, 9, 5), parity=0, chain=(10, 11)),
    pm.AtropBond(atoms=(8, 7, 3, 1, None, 5), parity=1),
    pm.AtropBond(atoms=(8, 7, 3, 1, 9, 5), parity=-1, chain=(10,)),
]
UNDETERMINED: list[LS] = [
    type(ls).model_validate({**dict(ls), "parity": None}) for ls in DETERMINED
]


def _rebuilt(ls: LS, perm: tuple[int, ...]) -> LS:
    """`ls` built again from its atom tuple permuted by `perm`."""
    data = {**dict(ls), "atoms": tuple(ls.atoms[i] for i in perm)}
    if isinstance(ls, pm.PlanarBond | pm.AtropBond) and perm[2] == 3:  # ends swap
        data["chain"] = ls.chain[::-1]
    return type(ls).model_validate(data)


@pytest.mark.parametrize("ls", DETERMINED + UNDETERMINED)
def test_normal_form_is_invariant_under_the_group(ls: LS) -> None:
    """Every permuted tuple builds the same LS: == compares configurations."""
    group = _GROUP if ls.parity is not None else _FULL_GROUP
    rebuilt = {_rebuilt(ls, perm) for perm in group[ls.kind]}
    assert rebuilt == {ls}
    assert {hash(other) for other in rebuilt} == {hash(ls)}


@pytest.mark.parametrize("ls", [ls for ls in DETERMINED if ls.parity])
def test_inverted_atoms_with_opposite_parity_are_equal(ls: LS) -> None:
    """The inversion with the parity negated describes the same configuration."""
    inverted = type(ls).model_validate(
        {
            **dict(ls),
            "atoms": tuple(ls.atoms[i] for i in _INVERSION[ls.kind]),
            "parity": -(ls.parity or 0),
        }
    )
    assert inverted == ls
    assert hash(inverted) == hash(ls)


def test_normal_form_sorts_none_first() -> None:
    """None sorts before every atom."""
    ls = pm.Tetrahedral(atoms=(0, 1, 2, 3, None))
    assert ls.atoms == (0, None, 1, 2, 3)


def test_normal_form_reverses_chain_with_the_ends() -> None:
    """The chain still runs from end1 to end2 after the ends swap."""
    bond = pm.PlanarBond(atoms=(8, 7, 3, 1, 9, 5), parity=0, chain=(10, 11))
    assert bond.atoms == (5, 9, 1, 3, 7, 8)
    assert bond.chain == (11, 10)


def test_atrop_bond_end_swap_keeps_the_helicity() -> None:
    """A chiral axis's C2 maps (a, b, e1, e2, c, d) onto (d, c, e2, e1, b, a).

    E.g. an allene with a = (1, 0, -1.5), b = (-1, 0, -1.5), e1 = (0, 0, -1),
    e2 = (0, 0, 1), c = (0, -1, 1.5), d = (0, 1, 1.5), turned by (x, y, z) → (y, x, -z).
    Swapping a and b in that image gives the mirror image.
    """
    ls = pm.AtropBond(atoms=(0, 1, 2, 3, 4, 5), parity=1)
    assert pm.AtropBond(atoms=(5, 4, 3, 2, 1, 0), parity=1) == ls
    assert pm.AtropBond(atoms=(4, 5, 3, 2, 0, 1), parity=1) == ls
    assert pm.AtropBond(atoms=(4, 5, 3, 2, 1, 0), parity=1) == pm.invert(ls)


@pytest.mark.parametrize("kind", sorted(_GROUP))
def test_groups_match_the_back_end(kind: str) -> None:
    """The permutation groups and inversions are stereomolgraph's (P14)."""
    from protomol._vendor.stereomolgraph import stereodescriptors  # noqa: PLC0415

    smg_cls = getattr(stereodescriptors, kind)
    assert set(_GROUP[kind]) == set(smg_cls.PERMUTATION_GROUP)
    assert _INVERSION.get(kind) == smg_cls.inversion


@pytest.mark.parametrize("ls", [ls for ls in DETERMINED if ls.kind != "SquarePlanar"])
def test_configurations_differ(ls: LS) -> None:
    """A determined LS and its inverse are not equal."""
    assert ls != pm.invert(ls)


def test_square_planar_is_its_own_inverse() -> None:
    """A square-planar centre is achiral: its mirror image is itself."""
    ls = pm.SquarePlanar(atoms=(0, 4, 2, 3, 1), parity=0)
    assert pm.invert(ls) == ls


@pytest.mark.parametrize("ls", DETERMINED + UNDETERMINED)
def test_invert_twice_is_identity(ls: LS) -> None:
    """Inverting twice gives the LS back, for all three kinds."""
    assert pm.invert(pm.invert(ls)) == ls


@pytest.mark.parametrize("ls", UNDETERMINED)
def test_invert_undetermined_is_unchanged(ls: LS) -> None:
    """An undetermined LS has no other configuration."""
    assert pm.invert(ls) == ls


@pytest.mark.parametrize("ls", DETERMINED + UNDETERMINED)
def test_isomers_are_distinct_and_closed_under_invert(ls: LS) -> None:
    """The isomers differ, invert maps them onto each other, and they keep the site."""
    isos = pm.isomers(ls)
    assert len(set(isos)) == len(isos)
    assert {pm.invert(iso) for iso in isos} == set(isos)
    assert all(pm.site(iso) == pm.site(ls) for iso in isos)


@pytest.mark.parametrize(
    ("ls", "count"),
    [
        (pm.Tetrahedral(atoms=(0, 1, 2, 3, 4)), 2),
        (pm.SquarePlanar(atoms=(0, 1, 2, 3, 4)), 3),
        (pm.TrigonalBipyramidal(atoms=(0, 1, 2, 3, 4, 5)), 20),
        (pm.Octahedral(atoms=(0, 1, 2, 3, 4, 5, 6)), 30),
        (pm.PlanarBond(atoms=(1, 2, 3, 4, 5, 6)), 2),
        (pm.AtropBond(atoms=(1, 2, 3, 4, 5, 6)), 2),
    ],
)
def test_isomer_counts(ls: LS, count: int) -> None:
    """The number of configurations with distinct atoms."""
    assert len(pm.isomers(ls)) == count


@pytest.mark.parametrize("ls", DETERMINED)
def test_isomers_order_is_fixed(ls: LS) -> None:
    """The order doesn't depend on the configuration of the input."""
    assert pm.isomers(ls) == pm.isomers(pm.invert(ls))
    assert ls in pm.isomers(ls)


def test_undetermined_has_one_normal_form() -> None:
    """Every shape-preserving arrangement of an undetermined LS builds one LS."""
    built = {
        pm.Tetrahedral(atoms=(0, *perm))
        for perm in itertools.permutations((4, 2, None, 1))
    }
    assert built == {pm.Tetrahedral(atoms=(0, None, 1, 2, 4))}
    assert next(iter(built)).atoms == (0, None, 1, 2, 4)


def test_site() -> None:
    """The centre atom, or the sorted end atoms."""
    assert pm.site(pm.Tetrahedral(atoms=(4, 1, 2, 3, None))) == 4
    assert pm.site(pm.AtropBond(atoms=(8, 7, 3, 1, 9, 5))) == (1, 3)


@pytest.mark.parametrize("chain", [(), (9, 10), (9, 10, 11, 12)])
def test_planar_bond_accepts_even_chain(chain: tuple[int, ...]) -> None:
    """A double bond or an odd cumulated chain."""
    assert pm.PlanarBond(atoms=(1, 2, 3, 4, 5, 6), chain=chain).chain == chain


@pytest.mark.parametrize("chain", [(9,), (9, 10, 11)])
def test_planar_bond_rejects_odd_chain(chain: tuple[int, ...]) -> None:
    """An odd number of inner atoms is an even chain, an AtropBond."""
    with pytest.raises(ValidationError, match="PlanarBond chain"):
        pm.PlanarBond(atoms=(1, 2, 3, 4, 5, 6), chain=chain)


@pytest.mark.parametrize("chain", [(), (9,), (9, 10, 11)])
def test_atrop_bond_accepts_empty_or_odd_chain(chain: tuple[int, ...]) -> None:
    """A rigid single bond or an even cumulated chain."""
    assert pm.AtropBond(atoms=(1, 2, 3, 4, 5, 6), chain=chain).chain == chain


@pytest.mark.parametrize("chain", [(9, 10)])
def test_atrop_bond_rejects_even_nonempty_chain(chain: tuple[int, ...]) -> None:
    """An even number of inner atoms is an odd chain, a PlanarBond."""
    with pytest.raises(ValidationError, match="AtropBond chain"):
        pm.AtropBond(atoms=(1, 2, 3, 4, 5, 6), chain=chain)


def test_stereo_change_rejects_two_sites() -> None:
    """All set slots share one site."""
    with pytest.raises(ValidationError, match="share one site"):
        pm.StereoChange(
            reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1),
            product=pm.Tetrahedral(atoms=(1, 0, 2, 3, 4), parity=1),
        )
    with pytest.raises(ValidationError, match="share one site"):
        pm.StereoChange(
            reactant=pm.PlanarBond(atoms=(5, 6, 1, 2, 7, 8), parity=0),
            ts=pm.Tetrahedral(atoms=(1, 2, 5, 6, None)),
        )


def test_stereo_change_rejects_no_slots() -> None:
    """At least one slot is set."""
    with pytest.raises(ValidationError, match="at least one slot"):
        pm.StereoChange()


def test_stereo_change_round_trips_json() -> None:
    """Bond LS of either kind and end order share their site; JSON round-trips."""
    change = pm.StereoChange(
        reactant=pm.PlanarBond(atoms=(5, 6, 1, 2, 7, 8), parity=0),
        ts=pm.AtropBond(atoms=(7, 8, 2, 1, 5, 6)),
    )
    assert pm.StereoChange.model_validate_json(change.model_dump_json()) == change
