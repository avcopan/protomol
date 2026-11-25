"""`low_t(g)`: the low-T scheme's sites on any graph (P54, P57).

The sites come from a perceived Lewis structure (`_lewis`) through the pipeline
`from_smiles` and `from_geometry` share (`_from_rdkit_mol`); the given graph's parities
are kept where it has the same site.
"""

from typing import Any, overload

from ._lewis import _lewis
from ._reaction import products, reactants
from ._smiles import _from_rdkit_mol
from ._stereo import AtropBond, PlanarBond, StereoChange, site
from ._types import (
    _LS,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
    _Graph,
    _site,
    _site_key,
    _stereo_fields,
)


@overload
def low_t(g: MolGraph | StereoMolGraph) -> StereoMolGraph: ...
@overload
def low_t(g: ReactionGraph | StereoReactionGraph) -> StereoReactionGraph: ...
def low_t(
    g: MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
) -> StereoMolGraph | StereoReactionGraph:
    """The graph with the low-T scheme's stereo sites (P54), keeping its parities.

    The sites are perceived on a Lewis structure of the graph. A site the graph already
    has keeps its LS and parity; a site the scheme drops loses its LS; a site it adds
    is undetermined, or fixed if it is trivial. No parity is invented, so a plain graph
    gets the sites with only the trivial ones fixed. `low_t` is idempotent, and leaves
    the output of `from_smiles` and `from_geometry` as it is, except where the rules
    depend on which Lewis structure is read (open question 11).

    A reaction graph's two sides are reassigned separately: each `StereoChange` slot
    follows its side's site, and a change left with no LS in any slot is removed. The
    TS slots are kept as they are (TS sites come with Phase 5).

    >>> from protomol import from_smiles, without_stereo
    >>> gra = from_smiles("C[C@H](N)CC")
    >>> low_t(without_stereo(gra)).atom_stereo[1].parity is None  # C1: undetermined
    True
    >>> low_t(gra) == gra
    True

    Raises:
        GraphError: If no Lewis structure fits the bonds, charge and spin.
    """
    if isinstance(g, MolGraph | StereoMolGraph):
        return _low_t_species(g)
    rct = _low_t_species(reactants(g))
    prd = _low_t_species(products(g))
    rls = {_site(ls): ls for ls in (*rct.atom_stereo, *rct.bond_stereo)}
    pls = {_site(ls): ls for ls in (*prd.atom_stereo, *prd.bond_stereo)}
    ts = (
        {_site(c): c.ts for c in g.stereo_changes if c.ts is not None}
        if isinstance(g, StereoReactionGraph)
        else {}
    )
    items: list[_LS | StereoChange] = []
    for key in sorted(rls.keys() | pls.keys() | ts.keys(), key=_site_key):
        r, p, t = rls.get(key), pls.get(key), ts.get(key)
        if r == p and t is None:
            assert r is not None  # noqa: S101 (for the type checker)
            items.append(r)
        else:
            items.append(StereoChange(reactant=r, product=p, ts=t))
    data: dict[str, Any] = {
        **_base_fields(g),
        "breaking": g.breaking,
        "forming": g.forming,
    }
    return StereoReactionGraph(**data, **_stereo_fields(items, reaction=True))


def _low_t_species(gra: MolGraph | StereoMolGraph) -> StereoMolGraph:
    """`low_t` of a species graph."""
    perceived = _from_rdkit_mol(_lewis(gra), charge=gra.charge, spin=gra.spin)
    if not isinstance(gra, StereoMolGraph):
        return perceived
    given = {site(ls): ls for ls in (*gra.atom_stereo, *gra.bond_stereo)}
    lss = [
        _kept(given.get(site(ls)), ls)
        for ls in (*perceived.atom_stereo, *perceived.bond_stereo)
    ]
    return StereoMolGraph(**_base_fields(gra), **_stereo_fields(lss, reaction=False))


def _base_fields(g: _Graph) -> dict[str, Any]:
    """The atoms, bonds, charge and spin of a graph."""
    return {name: getattr(g, name) for name in ("symbols", "bonds", "charge", "spin")}


def _kept(given: _LS | None, perceived: _LS) -> _LS:
    """The given LS if it is the perceived site's LS with a parity, else the perceived.

    The same site means the same kind on the same atoms (and chain): the given LS
    keeps its parity (and its atom order). A given LS without a parity gives way to
    the perceived one, which may be fixed as trivial.
    """
    if (
        given is None
        or given.parity is None
        or type(given) is not type(perceived)
        or set(given.atoms) != set(perceived.atoms)
    ):
        return perceived
    if isinstance(given, PlanarBond | AtropBond):
        assert isinstance(perceived, PlanarBond | AtropBond)  # noqa: S101
        if set(given.chain) != set(perceived.chain):
            return perceived
    return given
