"""Local stereodescriptors (LS): frozen models over stereomolgraph's atom tuples.

The atom tuples, parities and permutation groups mirror stereomolgraph's `Tetrahedral`,
`SquarePlanar`, `TrigonalBipyramidal`, `Octahedral`, `PlanarBond` and `AtropBond`
(`stereodescriptors.py`) (P14). Bond LS add a `chain` of
inner atoms for cumulated chains (P15).

Every LS is stored in normal form: the minimum atom tuple, with None before every atom,
over the arrangements that describe its configuration. For a chiral LS those are its
permutation group applied to its atoms with its parity, and to its inverted atoms with
the opposite parity (stereomolgraph's `inversion`), so the parity may change sign. So
`==` and `hash` compare configurations, as stereomolgraph's do (stereomol M3.1).
"""

import itertools
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator

type OInt = int | None  # None: a lone pair or a missing substituent


class Tetrahedral(BaseModel, frozen=True):
    """A tetrahedral centre: (center, n1, n2, n3, n4).

    Seen with n1 at the back, n2 → n3 → n4 turn clockwise for parity 1 and
    counter-clockwise for parity -1 (stereomolgraph's convention). None marks a lone
    pair.
    """

    kind: Literal["Tetrahedral"] = "Tetrahedral"
    atoms: tuple[int, OInt, OInt, OInt, OInt]
    parity: Literal[1, -1] | None = None

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        return _set_normal_form(self)


class SquarePlanar(BaseModel, frozen=True):
    """A square-planar centre: (center, n1, n2, n3, n4).

    The neighbours go in order round the square (n1 trans to n3, n2 trans to n4).
    The configuration is in the atom order, so a determined LS has parity 0. It is
    achiral: `invert` returns it unchanged.
    """

    kind: Literal["SquarePlanar"] = "SquarePlanar"
    atoms: tuple[int, OInt, OInt, OInt, OInt]
    parity: Literal[0] | None = None

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        return _set_normal_form(self)


class TrigonalBipyramidal(BaseModel, frozen=True):
    """A trigonal-bipyramidal centre: (center, a1, a2, e1, e2, e3), e.g. an SN2 TS.

    a1 and a2 are axial (top and bottom). With e1 turned to the back, e2 → e3 go
    round the equator in the direction the parity gives (stereomolgraph's convention
    and diagram).
    """

    kind: Literal["TrigonalBipyramidal"] = "TrigonalBipyramidal"
    atoms: tuple[int, OInt, OInt, OInt, OInt, OInt]
    parity: Literal[1, -1] | None = None

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        return _set_normal_form(self)


class Octahedral(BaseModel, frozen=True):
    """An octahedral centre: (center, n1, …, n6), n1 trans to n2, n3 to n5, n4 to n6.

    n3 → n4 → n5 → n6 go round the equator. Parity ±1 for the two handednesses, as in
    stereomolgraph (`Octahedral`).
    """

    kind: Literal["Octahedral"] = "Octahedral"
    atoms: tuple[int, OInt, OInt, OInt, OInt, OInt, OInt]
    parity: Literal[1, -1] | None = None

    @model_validator(mode="after")
    def _normalize(self) -> Self:
        return _set_normal_form(self)


class PlanarBond(BaseModel, frozen=True):
    """A planar bond or an odd cumulated chain: (a, b, end1, end2, c, d).

    a and b sit on end1, and c and d on end2. The configuration is in the atom order
    (a is cis to c), so a determined LS has parity 0. `chain` lists the inner atoms
    from end1 to end2; it is empty for an ordinary double bond.
    """

    kind: Literal["PlanarBond"] = "PlanarBond"
    atoms: tuple[OInt, OInt, int, int, OInt, OInt]
    parity: Literal[0] | None = None
    chain: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _check_chain_and_normalize(self) -> Self:
        if len(self.chain) % 2:
            msg = (
                "A PlanarBond chain needs an even number of inner atoms "
                f"(an odd number of cumulated double bonds), got {self.chain}"
            )
            raise ValueError(msg)
        return _set_normal_form(self)


class AtropBond(BaseModel, frozen=True):
    """A rigid single bond (atropisomer) or an even cumulated chain (allene axis).

    Same atom order as `PlanarBond`, with parity ±1 for the two helicities.
    """

    kind: Literal["AtropBond"] = "AtropBond"
    atoms: tuple[OInt, OInt, int, int, OInt, OInt]
    parity: Literal[1, -1] | None = None
    chain: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _check_chain_and_normalize(self) -> Self:
        if self.chain and not len(self.chain) % 2:
            msg = (
                "An AtropBond chain is empty (a single bond) or has an odd number of "
                "inner atoms (an even number of cumulated double bonds), "
                f"got {self.chain}"
            )
            raise ValueError(msg)
        return _set_normal_form(self)


type AtomStereo = Annotated[
    Tetrahedral | SquarePlanar | TrigonalBipyramidal | Octahedral,
    Field(discriminator="kind"),
]
type BondStereo = Annotated[PlanarBond | AtropBond, Field(discriminator="kind")]
type Stereo = Annotated[
    Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond,
    Field(discriminator="kind"),
]

# The atom-centred kinds, for isinstance
_ATOM_KINDS = (Tetrahedral, SquarePlanar, TrigonalBipyramidal, Octahedral)


class StereoChange(BaseModel, frozen=True):
    """One stereo site that changes in a reaction: its LS on each side and at the TS.

    None in a slot means no LS there (not a stereo element on that side).
    stereomolgraph calls the slots BROKEN, FORMED and FLEETING.
    """

    reactant: Stereo | None = None
    product: Stereo | None = None
    ts: Stereo | None = None

    @model_validator(mode="after")
    def _check_one_site(self) -> Self:
        slots = [ls for ls in (self.reactant, self.product, self.ts) if ls is not None]
        if not slots:
            msg = "A StereoChange needs an LS in at least one slot"
            raise ValueError(msg)
        sites = {site(ls) for ls in slots}
        if len(sites) > 1:
            msg = f"The LS of a StereoChange must share one site, got {sites}"
            raise ValueError(msg)
        return self


# Permutation groups of a determined LS, from stereomolgraph's `stereodescriptors.py`
# (StereoMolGraph `canon` branch, commit 2f94e9d). Each permutation lists the old
# positions in their new order.
_GROUP: dict[str, tuple[tuple[int, ...], ...]] = {
    "Tetrahedral": (
        (0, 1, 2, 3, 4),
        (0, 3, 1, 2, 4),
        (0, 2, 3, 1, 4),
        (0, 1, 4, 2, 3),
        (0, 2, 1, 4, 3),
        (0, 4, 2, 1, 3),
        (0, 1, 3, 4, 2),
        (0, 4, 1, 3, 2),
        (0, 3, 4, 1, 2),
        (0, 2, 4, 3, 1),
        (0, 3, 2, 4, 1),
        (0, 4, 3, 2, 1),
    ),
    "SquarePlanar": (
        (0, 1, 2, 3, 4),
        (0, 2, 3, 4, 1),
        (0, 3, 4, 1, 2),
        (0, 4, 1, 2, 3),
        (0, 4, 3, 2, 1),
        (0, 3, 2, 1, 4),
        (0, 2, 1, 4, 3),
        (0, 1, 4, 3, 2),
    ),
    "TrigonalBipyramidal": (
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 5, 3, 4),
        (0, 1, 2, 4, 5, 3),
        (0, 2, 1, 3, 5, 4),
        (0, 2, 1, 5, 4, 3),
        (0, 2, 1, 4, 3, 5),
    ),
    "Octahedral": (
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 6, 3, 4, 5),
        (0, 1, 2, 5, 6, 3, 4),
        (0, 1, 2, 4, 5, 6, 3),
        (0, 2, 1, 4, 3, 6, 5),
        (0, 2, 1, 5, 4, 3, 6),
        (0, 2, 1, 6, 5, 4, 3),
        (0, 2, 1, 3, 6, 5, 4),
        (0, 3, 5, 2, 4, 1, 6),
        (0, 3, 5, 6, 2, 4, 1),
        (0, 3, 5, 1, 6, 2, 4),
        (0, 3, 5, 4, 1, 6, 2),
        (0, 5, 3, 1, 4, 2, 6),
        (0, 5, 3, 6, 1, 4, 2),
        (0, 5, 3, 2, 6, 1, 4),
        (0, 5, 3, 4, 2, 6, 1),
        (0, 4, 6, 3, 2, 5, 1),
        (0, 4, 6, 1, 3, 2, 5),
        (0, 4, 6, 5, 1, 3, 2),
        (0, 4, 6, 2, 5, 1, 3),
        (0, 6, 4, 3, 1, 5, 2),
        (0, 6, 4, 2, 3, 1, 5),
        (0, 6, 4, 5, 2, 3, 1),
        (0, 6, 4, 1, 5, 2, 3),
    ),
    "PlanarBond": (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 0, 1),
        (5, 4, 3, 2, 1, 0),
    ),
    "AtropBond": (
        (0, 1, 2, 3, 4, 5),
        (1, 0, 2, 3, 5, 4),
        (4, 5, 3, 2, 1, 0),
        (5, 4, 3, 2, 0, 1),
    ),
}

# The atom permutation that, with the parity negated, describes the same configuration
# (stereomolgraph's `inversion`). An achiral kind (parity 0) has none.
_INVERSION: dict[str, tuple[int, ...]] = {
    "Tetrahedral": (0, 2, 1, 3, 4),
    "TrigonalBipyramidal": (0, 1, 2, 3, 5, 4),
    "Octahedral": (0, 2, 1, 3, 4, 5, 6),
    "AtropBond": (1, 0, 2, 3, 4, 5),
}

# The parity of a determined LS of each kind, before normalization.
_DETERMINED_PARITY: dict[str, int] = {
    "Tetrahedral": 1,
    "SquarePlanar": 0,
    "TrigonalBipyramidal": 1,
    "Octahedral": 1,
    "PlanarBond": 0,
    "AtropBond": 1,
}

# Permutation groups of an undetermined LS: every permutation that keeps its shape (the
# centre first; for a bond, {a, b} stays on its end, which may swap with the other).
_BOND_FULL_GROUP = tuple(
    perm
    for ab, cd in itertools.product(
        itertools.permutations((0, 1)), itertools.permutations((4, 5))
    )
    for perm in ((*ab, 2, 3, *cd), (*cd, 3, 2, *ab))
)
_FULL_GROUP: dict[str, tuple[tuple[int, ...], ...]] = {
    **{
        kind: tuple((0, *perm) for perm in itertools.permutations(range(1, n)))
        for kind, n in (
            ("Tetrahedral", 5),
            ("SquarePlanar", 5),
            ("TrigonalBipyramidal", 6),
            ("Octahedral", 7),
        )
    },
    "PlanarBond": _BOND_FULL_GROUP,
    "AtropBond": _BOND_FULL_GROUP,
}


def site(
    ls: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond,
) -> int | tuple[int, int]:
    """The centre atom, or the sorted pair of end atoms (the key in the graph's maps).

    >>> site(Tetrahedral(atoms=(0, 1, 2, 3, None)))
    0
    >>> site(PlanarBond(atoms=(5, 6, 3, 2, 7, 8)))
    (2, 3)
    """
    if isinstance(ls, _ATOM_KINDS):
        return ls.atoms[0]
    end1, end2 = ls.atoms[2], ls.atoms[3]
    return (end1, end2) if end1 < end2 else (end2, end1)


def invert[
    S: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
](ls: S) -> S:
    """The other configuration. An undetermined LS is returned unchanged.

    A chiral LS (parity ±1) gives its mirror image, by flipping the parity.
    `PlanarBond` swaps c and d (cis ⇄ trans). `SquarePlanar` is returned unchanged:
    it is its own mirror image, and it has three configurations, not two
    (`isomers` lists them).

    >>> invert(PlanarBond(atoms=(1, 2, 3, 4, 5, 6), parity=0)).atoms
    (1, 2, 3, 4, 6, 5)
    """
    if ls.parity is None:
        return ls
    if isinstance(ls, SquarePlanar):
        return ls
    if isinstance(ls, PlanarBond):
        return _replace(ls, atoms=_permute(ls.atoms, (0, 1, 2, 3, 5, 4)))
    return _replace(ls, parity=-ls.parity)


def isomers[
    S: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
](ls: S) -> tuple[S, ...]:
    """Every determined configuration of `ls`, in a fixed order.

    The order doesn't depend on the configuration of `ls`: by atom tuple, then parity 1
    before parity -1. With distinct atoms there are 2 for `Tetrahedral`, `PlanarBond`
    and `AtropBond`, 3 for `SquarePlanar`, 20 for `TrigonalBipyramidal` and 30 for
    `Octahedral`.

    >>> [iso.parity for iso in isomers(Tetrahedral(atoms=(0, 1, 2, 3, 4)))]
    [1, -1]
    """
    parity = _DETERMINED_PARITY[ls.kind]
    configurations = {_rearranged(ls, perm, parity) for perm in _FULL_GROUP[ls.kind]}
    return tuple(
        sorted(
            configurations, key=lambda iso: (_sort_key(iso.atoms), -(iso.parity or 0))
        )
    )


def is_determined(
    ls: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond,
) -> bool:
    """Whether the parity is set."""
    return ls.parity is not None


def _sort_key(atoms: tuple[OInt, ...]) -> tuple[tuple[bool, int], ...]:
    """Sort None before every atom."""
    return tuple((atom is not None, atom or 0) for atom in atoms)


def _permute[T: tuple[OInt, ...]](atoms: T, perm: tuple[int, ...]) -> T:
    return type(atoms)(atoms[i] for i in perm)


def _replace[
    S: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
](ls: S, **update: object) -> S:
    """A copy of `ls` with some fields replaced, validated (so normalized)."""
    return type(ls).model_validate({**dict(ls), **update})


def _rearranged[
    S: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
](ls: S, perm: tuple[int, ...], parity: int | None) -> S:
    """`ls` with its atoms permuted by `perm` and the given parity, validated."""
    update: dict[str, object] = {"atoms": _permute(ls.atoms, perm), "parity": parity}
    if isinstance(ls, PlanarBond | AtropBond) and _swaps_ends(ls, perm):
        update["chain"] = ls.chain[::-1]
    return _replace(ls, **update)


def _swaps_ends(
    ls: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond,
    perm: tuple[int, ...],
) -> bool:
    return isinstance(ls, PlanarBond | AtropBond) and perm[2] == 3  # noqa: PLR2004


def _set_normal_form[
    S: Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
](ls: S) -> S:
    """Put a just-validated LS in normal form, in place (used by the validators).

    A determined LS takes the minimum over its group, and a chiral one also over its
    group applied to the inverted atoms with the opposite parity. An undetermined one
    takes it over every permutation that keeps its shape, so that it has one normal
    form. A bond's chain is reversed when its ends swap.
    """
    arrangements: list[tuple[tuple[int, ...], int | None]]
    if not is_determined(ls):
        arrangements = [(perm, ls.parity) for perm in _FULL_GROUP[ls.kind]]
    else:
        arrangements = [(perm, ls.parity) for perm in _GROUP[ls.kind]]
        inversion = _INVERSION.get(ls.kind)
        if inversion is not None and ls.parity:
            arrangements += [
                (_permute(inversion, perm), -ls.parity) for perm in _GROUP[ls.kind]
            ]
    perm, parity = min(
        arrangements,
        key=lambda arr: (_sort_key(_permute(ls.atoms, arr[0])), arr[1] or 0),
    )
    # The model is frozen, and this runs inside its own validation
    object.__setattr__(ls, "atoms", _permute(ls.atoms, perm))
    object.__setattr__(ls, "parity", parity)
    if isinstance(ls, PlanarBond | AtropBond) and _swaps_ends(ls, perm):
        object.__setattr__(ls, "chain", ls.chain[::-1])
    return ls
