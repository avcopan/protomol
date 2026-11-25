"""The canonical form of a graph, and its canonical JSON (P17, P22, P46).

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

from ._backend import canon_numbering
from ._json import to_json
from ._reaction import relabel, reverse
from ._types import MolGraph, ReactionGraph, StereoMolGraph, StereoReactionGraph


def canonical_form[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    gra: G,
) -> tuple[G, tuple[int, ...]]:
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


def canonical_json[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    gra: G,
) -> str:
    """The JSON of the canonical graph: `to_json(canonical_form(gra)[0])`.

    The value AutoStorage stores for a graph identity (P47).

    >>> from protomol import from_smiles
    >>> canonical_json(from_smiles("OC")) == canonical_json(from_smiles("CO"))
    True
    """
    canonical, _ = canonical_form(gra)
    return to_json(canonical)


def canonical_direction[R: (ReactionGraph, StereoReactionGraph)](
    gra: R,
) -> tuple[R, bool]:
    """`gra` or its reverse with the smaller canonical JSON, and whether reversed.

    A reaction graph and its `reverse` describe one TS from the two sides, and their
    canonical forms differ. This picks one of them by comparing `canonical_json`, so
    the result is the same for both, up to atom numbering if the two are equal (a
    symmetric self-reaction, where a tie keeps `gra`). Atom indices are kept, so an
    atom map from `canonical_form` of the result applies to the input's atoms as
    they are. `canonical_form` itself stays faithful to its input.

    >>> from protomol import examples
    >>> ts = examples.abstraction_ts()
    >>> chosen, flipped = canonical_direction(ts)
    >>> chosen2, flipped2 = canonical_direction(reverse(ts))
    >>> chosen == chosen2, flipped != flipped2
    (True, True)
    """
    rev = reverse(gra)
    if canonical_json(rev) < canonical_json(gra):
        return rev, True
    return gra, False
