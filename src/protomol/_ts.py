"""TS sites: the reacting core of a reaction graph and the LS it touches (P67).

The reacting core is the atoms in a breaking or forming bond; for a channel with no
bond change, the atoms of its changing sites (a bond's two ends, or an inverting
centre). A TS site is an LS with an atom in the core: an atom LS centred there, or a
bond LS with an end there. It is the same "touched" test `ts_graphs` uses (P41).
"""

from ._types import ReactionGraph, StereoReactionGraph, _Site, _site, _site_key


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
    >>> (rotation,) = ts_graphs(from_smiles(r"C/C=C\C"), from_smiles("C/C=C/C"))
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
