"""Graph comparison up to atom numbering: `isomorphism` and `deduplicate` (P48).

Both run the back-end's VF2++ (`vf2pp_all_isomorphisms`), which compares the atoms,
the bonds, the breaking and forming bonds, the LS and the stereo changes. `charge` and
`spin` must be equal. `==` stays labelled equality (P21, P34).
"""

from collections import defaultdict
from collections.abc import Iterable

from ._backend import Matchable, matchable, vf2pp
from ._types import (
    GraphError,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
)


def isomorphism[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    g1: G, g2: G
) -> tuple[int, ...] | None:
    """An atom map `m` with `relabel(g1, m) == g2`, or None if there is none.

    Atom k of `g2` is atom `m[k]` of `g1` (the direction of `relabel` and of
    `canonical_form`'s `atom_map`). Test the result with `is not None`: two empty graphs
    give `()`. Both graphs must be of one type (`GraphError` otherwise).

    >>> from protomol import from_smiles, relabel
    >>> g1, g2 = from_smiles("OC"), from_smiles("CO")
    >>> m = isomorphism(g1, g2)
    >>> relabel(g1, m) == g2
    True
    >>> isomorphism(from_smiles("C[C@H](O)F"), from_smiles("C[C@@H](O)F")) is None
    True
    """
    _check_one_type((g1, g2))
    return _isomorphism(g1, g2, matchable(g1), matchable(g2))


def deduplicate[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    graphs: Iterable[G],
) -> list[G]:
    """The first graph of each isomorphism class, in input order.

    Graphs are bucketed by a fingerprint (stereo-aware colour refinement: equal for
    isomorphic graphs) and compared with `isomorphism` within a bucket. All graphs must
    be of one type (`GraphError` otherwise).

    >>> from protomol import from_smiles
    >>> kept = deduplicate([from_smiles("OC"), from_smiles("CC"), from_smiles("CO")])
    >>> [g.symbols for g in kept]
    [('O', 'C', 'H', 'H', 'H', 'H'), ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')]
    """
    graphs = list(graphs)
    _check_one_type(graphs)
    buckets: defaultdict[object, list[tuple[G, Matchable]]] = defaultdict(list)
    kept: list[G] = []
    for gra in graphs:
        match = matchable(gra)
        bucket = buckets[_fingerprint(gra, match)]
        if any(_isomorphism(other, gra, m, match) is not None for other, m in bucket):
            continue
        bucket.append((gra, match))
        kept.append(gra)
    return kept


def _fingerprint(
    gra: MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
    match: Matchable,
) -> object:
    """Equal for isomorphic graphs; may collide. Not a canonical form."""
    return (len(gra.symbols), gra.charge, gra.spin, match.fingerprint)


def _isomorphism(
    g1: MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
    g2: MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
    m1: Matchable,
    m2: Matchable,
) -> tuple[int, ...] | None:
    if (g1.charge, g1.spin) != (g2.charge, g2.spin):
        return None
    return vf2pp(m1, m2)


def _check_one_type(graphs: Iterable[object]) -> None:
    types = {type(gra) for gra in graphs}
    if len(types) > 1:
        names = ", ".join(sorted(t.__name__ for t in types))
        msg = f"Graphs of different types can't be compared: {names}"
        raise GraphError(msg)
