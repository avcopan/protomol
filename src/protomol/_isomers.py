"""`stereoisomers(g)`: a species graph's distinct determined stereoisomers (P65).

Every undetermined LS is expanded into its configurations (`isomers`), the determined
ones are kept, and the combinations are deduplicated up to atom numbering
(`deduplicate`). Reaction graphs (fleeting stereo) come with TS sites (P74).
"""

import itertools

from ._compare import deduplicate
from ._low_t import low_t
from ._stereo import isomers
from ._types import _LS, MolGraph, StereoMolGraph, with_stereo


def stereoisomers(g: MolGraph | StereoMolGraph) -> list[StereoMolGraph]:
    """The distinct determined stereoisomers of a species graph, mirror images too.

    A `MolGraph` first gets the low-T scheme's sites (`low_t`); a `StereoMolGraph` is
    expanded at the LS it has. Each undetermined LS takes every configuration, the
    determined ones are kept, and graphs equal up to atom numbering are given once,
    the first in the order of `isomers` (parity 1 first, the lowest site varying
    slowest). Atom indices are those of `g`. A determined graph gives itself.

    >>> from protomol import from_smiles, stereo_at
    >>> isos = stereoisomers(from_smiles("CC(O)CC"))  # butan-2-ol: (R) and (S)
    >>> [stereo_at(smg, 1).parity for smg in isos]
    [1, -1]
    >>> len(stereoisomers(from_smiles("CC(O)C(O)C")))  # two centres, one meso form
    3

    Raises:
        TypeError: If `g` is a reaction graph (Phase 5).
        GraphError: If `g` is a `MolGraph` with no Lewis structure.
    """
    if not isinstance(g, MolGraph | StereoMolGraph):
        msg = f"stereoisomers takes a species graph, got a {type(g).__name__}"
        raise TypeError(msg)
    smg = low_t(g) if isinstance(g, MolGraph) else g
    open_lss: list[_LS] = [
        ls for ls in (*smg.atom_stereo, *smg.bond_stereo) if ls.parity is None
    ]
    combos = itertools.product(*(isomers(ls) for ls in open_lss))
    return deduplicate(with_stereo(smg, *combo) for combo in combos)
