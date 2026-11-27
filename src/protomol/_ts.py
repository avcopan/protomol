"""TS sites: the reacting core of a reaction graph and the LS it touches (P67).

The reacting core is the atoms in a breaking or forming bond; for a channel with no
bond change, the atoms of its changing sites (a bond's two ends, or an inverting
centre). A TS site is an LS with an atom in the core: an atom LS centred there, or a
bond LS with an end there. It is the same "touched" test `ts_graphs` uses (P41).
"""

import itertools
import warnings

import numpy as np

from ._lewis import _lewis
from ._stereo import (
    AtropBond,
    OInt,
    PlanarBond,
    StereoChange,
    Tetrahedral,
    TrigonalBipyramidal,
    isomers,
    site,
)
from ._types import (
    _LS,
    MolGraph,
    ReactionGraph,
    StereoReactionGraph,
    _Site,
    _site,
    _site_key,
    with_stereo,
)
from ._vendor.stereomolgraph.coords import handedness


def _reacting_core(gra: ReactionGraph | StereoReactionGraph) -> frozenset[int]:
    """The atoms in a breaking or forming bond, else those of the changing sites.

    >>> from protomol import from_smiles, ts_graphs
    >>> sn2, _ = ts_graphs(from_smiles("CBr.[I-]"), from_smiles("CI.[Br-]"))
    >>> sorted(_reacting_core(sn2))
    [0, 1, 2]
    """
    core = {atom for bond in (*gra.breaking, *gra.forming) for atom in bond}
    if core or not isinstance(gra, StereoReactionGraph):
        return frozenset(core)
    return frozenset(
        atom for change in gra.stereo_changes for atom in _site_atoms(_site(change))
    )


def _ts_sites(gra: StereoReactionGraph) -> tuple[_Site, ...]:
    r"""The sites of the LS and changes with an atom in the reacting core, sorted.

    >>> from protomol import from_smiles, ts_graphs
    >>> rotation, _ = ts_graphs(from_smiles(r"C/C=C\C"), from_smiles("C/C=C/C"))
    >>> _ts_sites(rotation)
    ((1, 2),)
    """
    core = _reacting_core(gra)
    sites = {
        _site(item)
        for item in (*gra.atom_stereo, *gra.bond_stereo, *gra.stereo_changes)
    }
    return tuple(
        sorted(
            (s for s in sites if any(a in core for a in _site_atoms(s))),
            key=_site_key,
        )
    )


def _site_atoms(site: _Site) -> tuple[int, ...]:
    """The centre, or a bond's two ends."""
    return (site,) if isinstance(site, int) else site


def _ts_slot(change: StereoChange, gra: StereoReactionGraph) -> list[_LS | None]:
    """The `ts` slot configurations of a change at a TS site, one per channel (P70).

    The first rule that applies:
    - r and p equal: that LS.
    - One side only: its LS (still pyramidal or planar at the TS, or the attacked face
      or the forming double bond the TS fixes), unless the other side has a triple
      bond there: then the LS undetermined (a bifurcation: the configuration is set
      after the TS). Both directions give one TS slot.
    - A `Tetrahedral` centre that swaps one neighbour (or a lone pair) for another,
      breaking one bond and forming the other: with inversion, a
      `TrigonalBipyramidal` with the leaving and attacking atoms axial (backside);
      with retention, the `Tetrahedral` with None where the two meet (frontside, a
      1,2-shift, a proton leaving a lone pair; P71 as amended by P80); that
      `Tetrahedral` undetermined if a side is.
    - With no bond change, a `PlanarBond` (not a chain): an `AtropBond` on r's atoms,
      both parities (the mirror-image twisted TSs, P39); a lone-pair centre: None
      (planar at the TS).
    - Else: r undetermined, with a warning.
    A change with only a `ts` slot keeps it. Each rule gives the same slot for the
    reverse reaction, so a TS has one graph whichever way it was built (P81).
    """
    r, p = change.reactant, change.product
    if r is None and p is None:
        return [change.ts]
    if r == p:
        return [r]
    if r is None or p is None:
        return [_one_side_slot(r, p, gra)]
    slots: list[_LS | None] | None = None
    no_bond_change = not (gra.breaking or gra.forming)
    if isinstance(r, Tetrahedral) and isinstance(p, Tetrahedral):
        slots = (
            [None] if no_bond_change and None in r.atoms else _centre_slot(r, p, gra)
        )
    elif (
        isinstance(r, PlanarBond)
        and isinstance(p, PlanarBond)
        and no_bond_change
        and not r.chain
    ):
        slots = [*isomers(AtropBond(atoms=r.atoms))]
    if slots is not None:
        return slots
    msg = (
        f"No TS-slot rule for the change at {site(r)} from {r} to {p}: its ts slot is "
        "left undetermined"
    )
    warnings.warn(msg, stacklevel=2)
    return [_undetermined(r)]


def _one_side_slot(r: _LS | None, p: _LS | None, gra: StereoReactionGraph) -> _LS:
    """The slot of a change with an LS on one side only (`_ts_slot`)."""
    ls = r if p is None else p
    assert ls is not None  # noqa: S101 (the caller checks)
    other = "product" if p is None else "reactant"
    if isinstance(ls, PlanarBond | AtropBond) and _is_triple(gra, other, site(ls)):
        return _undetermined(ls)
    return ls


def _centre_slot(
    r: Tetrahedral, p: Tetrahedral, gra: StereoReactionGraph
) -> list[_LS | None] | None:
    """The slot of a centre that swaps a neighbour (`_ts_slot`), or None if it doesn't.

    Inversion gives the backside TBP; retention the `Tetrahedral` with None where the
    leaving and attacking atoms meet, the same from either direction.
    """
    exchange = _exchange(r, p, gra)
    if exchange is None:
        return None
    leaving, attacking = exchange
    shared = _swapped(r.atoms, leaving, None)
    if r.parity is None or p.parity is None:
        return [Tetrahedral(atoms=shared)]
    moved = Tetrahedral(atoms=_swapped(r.atoms, leaving, attacking), parity=r.parity)
    if moved == p:
        return [Tetrahedral(atoms=shared, parity=r.parity)]
    return [_trigonal_bipyramid(r, leaving, attacking)]


def _with_ts_slots(gra: StereoReactionGraph) -> list[StereoReactionGraph]:
    """`gra` with every combination of `ts` slots at its TS sites (`_ts_slot`)."""
    sites = set(_ts_sites(gra))
    changes = [c for c in gra.stereo_changes if _site(c) in sites]
    options = [
        [
            StereoChange(reactant=c.reactant, product=c.product, ts=ts)
            for ts in _ts_slot(c, gra)
        ]
        for c in changes
    ]
    return [with_stereo(gra, *combo) for combo in itertools.product(*options)]


def _with_ts_slot(gra: StereoReactionGraph) -> StereoReactionGraph:
    """`gra` with one `ts` slot at each TS site, for a builder that returns one graph.

    A slot keeps its LS if the rules allow it; with several configurations (the two
    twisted TSs of a rotation) the slot is their LS undetermined, an unchosen element.
    """
    sites = set(_ts_sites(gra))
    changes: list[StereoChange] = []
    for c in gra.stereo_changes:
        if _site(c) not in sites:
            continue
        options = _ts_slot(c, gra)
        if c.ts in options:
            ts = c.ts
        elif len(options) == 1:
            ts = options[0]
        else:
            first = options[0]
            assert first is not None  # noqa: S101 (several options are LS)
            ts = _undetermined(first)
        changes.append(StereoChange(reactant=c.reactant, product=c.product, ts=ts))
    return with_stereo(gra, *changes)


def _exchange(
    r: Tetrahedral, p: Tetrahedral, gra: StereoReactionGraph
) -> tuple[OInt, OInt] | None:
    """The (leaving, attacking) neighbours of a centre that swaps one, or None.

    Either may be a lone pair (None); an atom's bond to the centre must break (the
    leaving one) or form (the attacking one).
    """
    center = r.atoms[0]
    lost = set(r.atoms[1:]) - set(p.atoms[1:])
    gained = set(p.atoms[1:]) - set(r.atoms[1:])
    if len(lost) != 1 or len(gained) != 1:
        return None
    (leaving,), (attacking,) = lost, gained
    breaking = {frozenset(b) for b in gra.breaking}
    forming = {frozenset(b) for b in gra.forming}
    if (leaving is None or {center, leaving} in breaking) and (
        attacking is None or {center, attacking} in forming
    ):
        return leaving, attacking
    return None


def _swapped(atoms: tuple[OInt, ...], old: OInt, new: OInt) -> tuple[OInt, ...]:
    return tuple(new if a == old else a for a in atoms)


# A regular tetrahedron's corners, with stereomolgraph's `handedness` 1
_CORNERS = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]])


def _trigonal_bipyramid(
    tet: Tetrahedral, leaving: OInt, attacking: OInt
) -> TrigonalBipyramidal:
    """The backside TS of a `Tetrahedral`: leaving and attacking atoms axial.

    The three other neighbours flatten into the equator and keep their turn as seen
    from the leaving atom, which stays on its side; the attacking atom comes from the
    other. The parity is read on ideal coordinates as stereomolgraph's
    `_trigonal_bipyramidal_from_coords` reads it, so a TS geometry gives the same LS.
    """
    center, *nbrs = tet.atoms
    xyz = _ideal_tetrahedron(tet.parity)
    k = nbrs.index(leaving)  # None: the lone pair
    axis = xyz[k]
    others = [i for i in range(4) if i != k]
    equator = np.array(
        [xyz[i] - (xyz[i] @ axis) / (axis @ axis) * axis for i in others]
    )
    parity = -int(handedness(np.array([*equator, axis])))
    return TrigonalBipyramidal(
        atoms=(center, leaving, attacking, *(nbrs[i] for i in others)),
        parity=1 if parity > 0 else -1,
    )


def _ideal_tetrahedron(parity: int | None) -> np.ndarray:
    """Four corners for n1…n4 whose `handedness` is the parity (1 if None)."""
    fourth = np.array([[-1.0, -1.0, 1.0]])
    xyz = np.concatenate([_CORNERS, fourth])
    if int(handedness(xyz)) != (parity or 1):
        xyz = xyz[[0, 1, 3, 2]]
    return xyz


def _is_triple(gra: StereoReactionGraph, side: str, bond: _Site) -> bool:
    """Whether a Lewis structure of one side has a triple bond between the two atoms."""
    if isinstance(bond, int):
        return False
    dropped = set(gra.forming if side == "reactant" else gra.breaking)
    i, j = bond
    if (i, j) in dropped or (i, j) not in set(gra.bonds):
        return False
    mol = _lewis(
        MolGraph(
            symbols=gra.symbols,
            bonds=[b for b in gra.bonds if b not in dropped],
            charge=gra.charge,
            spin=gra.spin,
        )
    )
    return mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() == 3  # noqa: PLR2004


def _undetermined[S: _LS](ls: S) -> S:
    """The LS with its parity unset (same kind, atoms and chain)."""
    return type(ls).model_validate({**dict(ls), "parity": None})
