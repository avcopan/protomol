"""The canonical form of a graph, its canonical JSON (P17, P22, P46), and mirror images.

The numbering is the back-end's `canon_atom_num` (stereomolgraph's `canon` branch),
which sees the atoms, the bonds, the breaking and forming bonds, the LS and the stereo
changes. Any two canonical numberings of one graph differ by an automorphism, which
leaves the relabelled graph unchanged, so the canonical graph and its JSON depend only
on the graph, not on how its atoms are numbered. `charge` and `spin` are carried over as
they are.

A chain LS (P15) is kept in the canonical graph but not seen by the numbering, since
stereomolgraph can't hold one (P45): for a graph with one, the canonical form may
depend on the input numbering.
"""

from typing import Any, overload

from ._backend import canon_numbering
from ._json import to_json
from ._reaction import relabel, reverse
from ._stereo import (
    AtropBond,
    Octahedral,
    StereoChange,
    Tetrahedral,
    TrigonalBipyramidal,
    invert,
)
from ._types import (
    _LS,
    AnyGraph,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
    _fields,
)


# The precise overload keeps the type (`MolGraph` in, `MolGraph` out); the `AnyGraph`
# one accepts a union, which a constrained type variable rejects (P59).
@overload
def canonical_form[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    gra: G,
) -> tuple[G, tuple[int, ...]]: ...
@overload
def canonical_form(gra: AnyGraph) -> tuple[AnyGraph, tuple[int, ...]]: ...
def canonical_form(gra: AnyGraph) -> tuple[AnyGraph, tuple[int, ...]]:
    """The canonical graph and the atom map to it: `(canonical, atom_map)`.

    Atom k of `canonical` is atom `atom_map[k]` of `gra` (AutoStorage's `atom_map`
    direction, P22), so `canonical == relabel(gra, atom_map)`. Two graphs without chain
    LS are the same up to atom numbering exactly when their canonical graphs are equal.

    >>> from protomol import from_smiles
    >>> canonical, atom_map = canonical_form(from_smiles("OC"))
    >>> canonical.symbols, atom_map
    (('C', 'O', 'H', 'H', 'H', 'H'), (1, 0, 2, 3, 4, 5))
    >>> canonical == canonical_form(from_smiles("CO"))[0]
    True
    """
    atom_map = canon_numbering(gra)
    return relabel(gra, atom_map), atom_map


def canonical_json(gra: AnyGraph) -> str:
    """The JSON of the canonical graph: `to_json(canonical_form(gra)[0])`.

    The value AutoStorage stores for a graph identity (P47).

    >>> from protomol import from_smiles
    >>> canonical_json(from_smiles("OC")) == canonical_json(from_smiles("CO"))
    True
    """
    canonical, _ = canonical_form(gra)
    return to_json(canonical)


@overload
def canonical_direction[R: (ReactionGraph, StereoReactionGraph)](
    gra: R,
) -> tuple[R, bool]: ...
@overload
def canonical_direction(
    gra: ReactionGraph | StereoReactionGraph,
) -> tuple[ReactionGraph | StereoReactionGraph, bool]: ...
def canonical_direction(
    gra: ReactionGraph | StereoReactionGraph,
) -> tuple[ReactionGraph | StereoReactionGraph, bool]:
    """`gra` or its reverse with the smaller canonical JSON, and whether reversed.

    A reaction graph and its `reverse` describe one TS from the two sides, and their
    canonical forms differ. This picks one of them by comparing `canonical_json`, so
    the result is the same for both, up to atom numbering if the two are equal (a
    symmetric self-reaction, where a tie keeps `gra`). Atom indices are kept, so an
    atom map from `canonical_form` of the result applies to the input's atoms as
    they are. `canonical_form` itself stays faithful to its input.

    The canonical TS, the same for a TS, its reverse and their mirror images (P60),
    is `canonical_direction(canonical_enantiomer(gra))`.

    >>> from protomol import examples
    >>> ts = examples.abstraction_ts()
    >>> chosen, flipped = canonical_direction(ts)
    >>> chosen2, flipped2 = canonical_direction(reverse(ts))
    >>> chosen == chosen2, flipped != flipped2
    (True, True)
    """
    # Narrowed per type: `reverse`'s type variable rejects the union
    rev: ReactionGraph | StereoReactionGraph
    if isinstance(gra, ReactionGraph):  # noqa: SIM108
        rev = reverse(gra)
    else:
        rev = reverse(gra)
    if canonical_json(rev) < canonical_json(gra):
        return rev, True
    return gra, False


# Mirror images (P50, P60)

# The kinds whose determined LS has a mirror image other than itself
_CHIRAL_KINDS = (Tetrahedral, TrigonalBipyramidal, Octahedral, AtropBond)


@overload
def enantiomer[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    gra: G,
) -> G: ...
@overload
def enantiomer(gra: AnyGraph) -> AnyGraph: ...
def enantiomer(gra: AnyGraph) -> AnyGraph:
    """The mirror image: every chiral LS inverted, atom indices kept.

    `Tetrahedral`, `AtropBond`, `TrigonalBipyramidal` and `Octahedral` LS with a
    parity flip it, and so do the slots of a reaction graph's stereo changes;
    `PlanarBond` and `SquarePlanar` (their own mirror images) and undetermined LS are
    kept as they are. A graph without stereo fields is returned unchanged. Being
    index-aligned, the result needs no atom map: the geometry reflected through a
    plane is a geometry of it. A trivial site's fixed LS (a CH3 carbon's) is inverted
    too, which describes the same configuration, so compare results by canonical JSON.

    >>> from protomol import from_smiles
    >>> gra = from_smiles("C[C@H](O)/C=C/C")
    >>> canonical_json(enantiomer(gra)) == canonical_json(
    ...     from_smiles("C[C@@H](O)/C=C/C")
    ... )
    True
    >>> enantiomer(enantiomer(gra)) == gra
    True
    """
    if not isinstance(gra, StereoMolGraph | StereoReactionGraph):
        return gra
    data: dict[str, Any] = _fields(gra)
    data["atom_stereo"] = [_mirror_ls(ls) for ls in gra.atom_stereo]
    data["bond_stereo"] = [_mirror_ls(ls) for ls in gra.bond_stereo]
    if isinstance(gra, StereoReactionGraph):
        data["stereo_changes"] = [
            StereoChange(
                reactant=_mirror_slot(c.reactant),
                product=_mirror_slot(c.product),
                ts=_mirror_slot(c.ts),
            )
            for c in gra.stereo_changes
        ]
    return type(gra)(**data)


def is_chiral(gra: AnyGraph) -> bool:
    """Whether the graph differs from its mirror image, up to atom numbering.

    `canonical_json(enantiomer(gra)) != canonical_json(gra)`. A reaction graph is
    compared as it is directed.

    >>> from protomol import from_smiles
    >>> is_chiral(from_smiles("C[C@H](O)CC")), is_chiral(from_smiles("C[C@H](O)C"))
    (True, False)
    >>> is_chiral(from_smiles("C[C@@H](O)[C@@H](O)C"))  # meso
    False
    """
    return canonical_json(enantiomer(gra)) != canonical_json(gra)


@overload
def canonical_enantiomer[
    G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)
](gra: G) -> G: ...
@overload
def canonical_enantiomer(gra: AnyGraph) -> AnyGraph: ...
def canonical_enantiomer(gra: AnyGraph) -> AnyGraph:
    """`gra` or its `enantiomer`, whichever has the smaller canonical JSON.

    A reproducible choice of one of two mirror images (like InChI's `/m0`); a tie
    (an achiral graph) keeps `gra`. Atom indices are kept. On a reaction graph the
    choice doesn't depend on its direction (P60): each of the two is compared in its
    canonical direction, `canonical_json(canonical_direction(x)[0])`, and the winner
    is returned in `gra`'s direction.

    >>> from protomol import from_smiles
    >>> r, s = from_smiles("C[C@H](O)CC"), from_smiles("C[C@@H](O)CC")
    >>> canonical_json(canonical_enantiomer(r)) == canonical_json(s)
    True
    >>> canonical_enantiomer(s) == s
    True
    """
    mirror = enantiomer(gra)
    if _mirror_key(mirror) < _mirror_key(gra):
        return mirror
    return gra


def _mirror_key(gra: AnyGraph) -> str:
    """The canonical JSON, in the canonical direction for a reaction graph (P60)."""
    if isinstance(gra, ReactionGraph | StereoReactionGraph):
        return canonical_json(canonical_direction(gra)[0])
    return canonical_json(gra)


def _mirror_ls[S: _LS](ls: S) -> S:
    return invert(ls) if isinstance(ls, _CHIRAL_KINDS) else ls


def _mirror_slot(ls: _LS | None) -> _LS | None:
    return None if ls is None else _mirror_ls(ls)
