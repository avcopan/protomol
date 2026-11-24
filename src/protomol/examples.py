"""Test graphs: a species, an abstraction TS, a stereoisomerization TS.

Each is built from code; `tests/data/examples/<name>.json` holds its golden JSON.
"""

from typing import cast

from ._ccv import ts_graphs
from ._reaction import reaction_graph
from ._smiles import from_smiles
from ._stereo import AtropBond, PlanarBond, StereoChange
from ._types import StereoMolGraph, StereoReactionGraph, stereo_at, with_stereo


def species() -> StereoMolGraph:
    """(E)-pent-3-en-2-ol, one enantiomer: C[C@@H](O)/C=C/C (16 atoms, singlet).

    A tetrahedral centre, a planar bond, and four rotors (2 CH3, OH, C-C).

    >>> gra = species()
    >>> len(gra.symbols), gra.spin
    (16, 0)
    """
    return from_smiles("C[C@@H](O)/C=C/C")


def abstraction_ts() -> StereoReactionGraph:
    """C2H6 + OH → C2H5 + H2O (10 atoms, doublet): H abstraction, one channel.

    A CH3 rotor on each side of the reacting core (open question 4).

    >>> ts = abstraction_ts()
    >>> len(ts.symbols), ts.spin
    (10, 1)
    """
    (ts,) = ts_graphs(from_smiles("CC.[OH]"), from_smiles("C[CH2].O"))
    return ts


def stereoisomerization_ts() -> StereoReactionGraph:
    """(Z) → (E) but-2-en-1-yl radical (11 atoms, doublet): rotation about C2=C3.

    The C2=C3 planar bond comes from radical resonance, so this is a low-T stereo
    element (high-T, i.e. InChI, drops it). No bonds change; only that PlanarBond does.
    The twisted TS is a chiral axis, and its two mirror images are two channels (P39):
    this is the one with `AtropBond` parity 1.

    >>> ts = stereoisomerization_ts()
    >>> len(ts.symbols), ts.spin, ts.breaking, ts.forming
    (11, 1, (), ())
    """
    g = reaction_graph(from_smiles(r"[CH2]/C=C\C"), from_smiles(r"[CH2]/C=C/C"))
    change = cast("StereoChange", stereo_at(g, (1, 2)))
    z, e = cast("PlanarBond", change.reactant), change.product
    twist = AtropBond(atoms=z.atoms, parity=1)
    return with_stereo(g, StereoChange(reactant=z, product=e, ts=twist))
