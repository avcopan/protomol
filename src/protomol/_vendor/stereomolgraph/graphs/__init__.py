# ruff: noqa: F401
"""Simple access of key classes MolGraph, StereoMolGraph,
CondensedReactionGraph and StereoCondensedReactionGraph"""

from __future__ import annotations

__all__ = (
    "AtomId",
    "Bond",
    "MolGraph",
    "StereoMolGraph",
    "CondensedReactionGraph",
    "StereoCondensedReactionGraph",
)

from protomol._vendor.stereomolgraph.graphs.crg import Change, CondensedReactionGraph
from protomol._vendor.stereomolgraph.graphs.mg import AtomId, Bond, MolGraph
from protomol._vendor.stereomolgraph.graphs.scrg import StereoCondensedReactionGraph
from protomol._vendor.stereomolgraph.graphs.smg import StereoMolGraph
