"""`stereoisomers(g)` and `transition_state(srg)` (P65, P74).

Every undetermined LS is expanded into its configurations (`isomers`), the determined
ones are kept, and the combinations are deduplicated up to atom numbering
(`deduplicate`). On a reaction graph the `ts` slots follow the sides by the TS-slot
rules (P70), except a bifurcation's, which stays undetermined.
"""

import itertools
from typing import overload

from ._compare import deduplicate
from ._low_t import low_t
from ._stereo import StereoChange, isomers
from ._ts import _is_open_by_rule, _undetermined, _with_ts_slots
from ._types import (
    _LS,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
    with_stereo,
)


@overload
def stereoisomers(g: MolGraph | StereoMolGraph) -> list[StereoMolGraph]: ...
@overload
def stereoisomers(
    g: ReactionGraph | StereoReactionGraph,
) -> list[StereoReactionGraph]: ...
def stereoisomers(
    g: MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
) -> list[StereoMolGraph] | list[StereoReactionGraph]:
    """The distinct determined stereoisomers of a graph, mirror images too.

    A `MolGraph` or `ReactionGraph` first gets the low-T scheme's sites (`low_t`,
    which also sets a reaction graph's `ts` slots); a stereo graph is expanded at the
    LS it has. Each undetermined LS takes every configuration, the determined ones are
    kept, and graphs equal up to atom numbering are given once, the first in the order
    of `isomers` (parity 1 first, the lowest site varying slowest). Atom indices are
    those of `g`. A determined graph gives itself.

    On a reaction graph each change's side slots are expanded, then its `ts` slot is
    set by the TS-slot rules (P70; a determined slot the rules allow is kept), so a
    rotation gives its two twisted TSs. A bifurcation's `ts` slot stays undetermined:
    one graph per product configuration, the channels (P74); `transition_state` gives
    their shared TS.

    >>> from protomol import from_smiles, stereo_at
    >>> isos = stereoisomers(from_smiles("CC(O)CC"))  # butan-2-ol: (R) and (S)
    >>> [stereo_at(smg, 1).parity for smg in isos]
    [1, -1]
    >>> len(stereoisomers(from_smiles("CC(O)C(O)C")))  # two centres, one meso form
    3

    Raises:
        TypeError: If `g` is not a graph.
        GraphError: If `g` is a `MolGraph` or `ReactionGraph` with no Lewis
            structure.
    """
    if isinstance(g, MolGraph | StereoMolGraph):
        smg = low_t(g) if isinstance(g, MolGraph) else g
        combos = itertools.product(*(isomers(ls) for ls in _open(smg)))
        return deduplicate(with_stereo(smg, *combo) for combo in combos)
    if not isinstance(g, ReactionGraph | StereoReactionGraph):
        msg = f"stereoisomers takes a graph, got {g!r}"
        raise TypeError(msg)
    srg = low_t(g) if isinstance(g, ReactionGraph) else g
    sides = [_side_configurations(c) for c in srg.stereo_changes]
    unchanged = itertools.product(*(isomers(ls) for ls in _open(srg)))
    found: list[StereoReactionGraph] = []
    for lss, changes in itertools.product(unchanged, itertools.product(*sides)):
        for cand in _with_ts_slots(with_stereo(srg, *lss, *changes)):
            found.extend(_expanded_ts_slots(cand))
    return deduplicate(found)


def transition_state(srg: StereoReactionGraph) -> StereoReactionGraph:
    r"""The TS of a channel: the side slots its TS doesn't fix made undetermined.

    Where the TS-slot rules leave a change's `ts` slot undetermined with its sides
    determined (a bifurcation: OH + 1-fluoropropyne makes a (Z) and an (E) alkenyl
    radical through one TS), the side slots become undetermined too, so the
    bifurcation's channels give one graph (P74, P5). Every other graph is returned as
    it is. A TS's stored identity is
    `canonical_direction(canonical_enantiomer(transition_state(g)))` (P60).

    >>> from protomol import from_smiles, ts_graphs
    >>> rct = from_smiles("CC#CF.[OH]")
    >>> (z,) = ts_graphs(rct, from_smiles(r"C/C(O)=[C]\F"))
    >>> (e,) = ts_graphs(rct, from_smiles(r"C/C(O)=[C]/F"))
    >>> z == e, transition_state(z) == transition_state(e)
    (False, True)

    Raises:
        TypeError: If `srg` is not a `StereoReactionGraph`.
    """
    if not isinstance(srg, StereoReactionGraph):
        msg = f"transition_state takes a StereoReactionGraph, got {srg!r}"
        raise TypeError(msg)
    changes = [
        StereoChange(
            reactant=None if c.reactant is None else _undetermined(c.reactant),
            product=None if c.product is None else _undetermined(c.product),
            ts=c.ts,
        )
        for c in srg.stereo_changes
        if c.ts is not None and c.ts.parity is None and _is_open_by_rule(c, srg)
    ]
    return with_stereo(srg, *changes)


def _open(g: StereoMolGraph | StereoReactionGraph) -> list[_LS]:
    """The undetermined unchanged LS."""
    return [ls for ls in (*g.atom_stereo, *g.bond_stereo) if ls.parity is None]


def _side_configurations(change: StereoChange) -> list[StereoChange]:
    """Every configuration of a change's side slots, the `ts` slot kept for now."""
    sides = [
        [ls] if ls is None or ls.parity is not None else list(isomers(ls))
        for ls in (change.reactant, change.product)
    ]
    return [
        StereoChange(reactant=r, product=p, ts=change.ts)
        for r, p in itertools.product(*sides)
    ]


def _expanded_ts_slots(srg: StereoReactionGraph) -> list[StereoReactionGraph]:
    """Every configuration of each undetermined `ts` slot but a bifurcation's."""
    options = [
        [
            StereoChange(reactant=c.reactant, product=c.product, ts=ts)
            for ts in isomers(c.ts)
        ]
        for c in srg.stereo_changes
        if c.ts is not None and c.ts.parity is None and not _is_open_by_rule(c, srg)
    ]
    return [with_stereo(srg, *combo) for combo in itertools.product(*options)]
