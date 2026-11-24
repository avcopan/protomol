"""Reaction graphs: building one from two sides, taking it apart, and relabelling.

Every function builds its result through a constructor (P36), so the result is
normalized and validated.
"""

from collections.abc import Sequence
from typing import Any, overload

from ._stereo import AtropBond, PlanarBond, StereoChange
from ._types import (
    _LS,
    GraphError,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
    _fields,
    _site,
    _site_key,
    _stereo_fields,
)

_STEREO_FIELDS = ("atom_stereo", "bond_stereo", "stereo_changes")


@overload
def reaction_graph(
    reactant: StereoMolGraph, product: StereoMolGraph
) -> StereoReactionGraph: ...
@overload
def reaction_graph(reactant: MolGraph, product: MolGraph) -> ReactionGraph: ...
def reaction_graph(
    reactant: MolGraph | StereoMolGraph, product: MolGraph | StereoMolGraph
) -> ReactionGraph | StereoReactionGraph:
    """The reaction graph of two index-aligned graphs (same symbols, charge and spin).

    Bonds only in the reactant are breaking, and bonds only in the product are forming.
    Equal LS stay unchanged (`atom_stereo`, `bond_stereo`); others become a
    `StereoChange(reactant=, product=)` in `stereo_changes`, with the TS slot empty
    (fleeting stereo comes in Phase 5).

    >>> r = MolGraph(symbols=["H", "H", "H"], bonds=[(0, 1)])
    >>> p = MolGraph(symbols=["H", "H", "H"], bonds=[(1, 2)])
    >>> g = reaction_graph(r, p)
    >>> g.bonds, g.breaking, g.forming
    (((0, 1), (1, 2)), ((0, 1),), ((1, 2),))

    Raises:
        TypeError: If the two graphs are not both `MolGraph`s or both
            `StereoMolGraph`s.
        GraphError: If their symbols, charge or spin differ.
    """
    if not (
        type(reactant) is type(product)
        and isinstance(reactant, MolGraph | StereoMolGraph)
    ):
        msg = (
            "Expected two MolGraphs or two StereoMolGraphs, got "
            f"{type(reactant).__name__} and {type(product).__name__}"
        )
        raise TypeError(msg)
    for name in ("symbols", "charge", "spin"):
        if getattr(reactant, name) != getattr(product, name):
            msg = (
                f"The two sides differ in {name}: {getattr(reactant, name)} and "
                f"{getattr(product, name)}"
            )
            raise GraphError(msg)
    rbonds, pbonds = set(reactant.bonds), set(product.bonds)
    data: dict[str, Any] = {
        "symbols": reactant.symbols,
        "bonds": rbonds | pbonds,
        "charge": reactant.charge,
        "spin": reactant.spin,
        "breaking": rbonds - pbonds,
        "forming": pbonds - rbonds,
    }
    if isinstance(reactant, MolGraph):
        return ReactionGraph(**data)
    assert isinstance(product, StereoMolGraph)  # noqa: S101 (for the type checker)
    rls = {_site(ls): ls for ls in (*reactant.atom_stereo, *reactant.bond_stereo)}
    pls = {_site(ls): ls for ls in (*product.atom_stereo, *product.bond_stereo)}
    items: list[_LS | StereoChange] = []
    for site in sorted(rls.keys() | pls.keys(), key=_site_key):
        r, p = rls.get(site), pls.get(site)
        if r == p:
            assert r is not None  # noqa: S101 (for the type checker)
            items.append(r)
        else:
            items.append(StereoChange(reactant=r, product=p))
    return StereoReactionGraph(**data, **_stereo_fields(items, reaction=True))


@overload
def reactants(gra: StereoReactionGraph) -> StereoMolGraph: ...
@overload
def reactants(gra: ReactionGraph) -> MolGraph: ...
def reactants(
    gra: ReactionGraph | StereoReactionGraph,
) -> MolGraph | StereoMolGraph:
    """The reactant side, as one (possibly disconnected) graph in the same atom order.

    Its bonds are `bonds` without `forming`, and its LS are the unchanged LS and the
    `reactant` slot of each change.

    >>> g = ReactionGraph(
    ...     symbols=["H", "H", "H"], bonds=[(0, 1), (1, 2)], breaking=[(0, 1)],
    ...     forming=[(1, 2)],
    ... )
    >>> reactants(g).bonds
    ((0, 1),)
    """
    return _side(gra, "reactant")


@overload
def products(gra: StereoReactionGraph) -> StereoMolGraph: ...
@overload
def products(gra: ReactionGraph) -> MolGraph: ...
def products(
    gra: ReactionGraph | StereoReactionGraph,
) -> MolGraph | StereoMolGraph:
    """The product side, as one (possibly disconnected) graph in the same atom order.

    Its bonds are `bonds` without `breaking`, and its LS are the unchanged LS and the
    `product` slot of each change.

    >>> g = ReactionGraph(
    ...     symbols=["H", "H", "H"], bonds=[(0, 1), (1, 2)], breaking=[(0, 1)],
    ...     forming=[(1, 2)],
    ... )
    >>> products(g).bonds
    ((1, 2),)
    """
    return _side(gra, "product")


def reverse[R: (ReactionGraph, StereoReactionGraph)](gra: R) -> R:
    """The reverse reaction: breaking and forming swap, and so do the change slots.

    The TS slot of each change stays as it is.

    >>> g = ReactionGraph(
    ...     symbols=["H", "H", "H"], bonds=[(0, 1), (1, 2)], breaking=[(0, 1)],
    ...     forming=[(1, 2)],
    ... )
    >>> reverse(g).breaking
    ((1, 2),)
    """
    data: dict[str, Any] = _fields(gra)
    data |= {"breaking": gra.forming, "forming": gra.breaking}
    if isinstance(gra, StereoReactionGraph):
        data["stereo_changes"] = [
            StereoChange(reactant=c.product, product=c.reactant, ts=c.ts)
            for c in gra.stereo_changes
        ]
    return type(gra)(**data)


def relabel[G: (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)](
    gra: G, atom_map: Sequence[int]
) -> G:
    """The graph with atom k of the result = atom `atom_map[k]` of `gra`.

    Same direction as AutoStorage's link `atom_map` (P22), whose `atom_map[k]` is the
    input index of canonical atom k: `relabel(gra, atom_map)` gives the canonical
    graph, and the inverse map (`numpy.argsort(atom_map)`) gives `gra` back.

    >>> g = MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)])
    >>> relabel(g, [2, 0, 1])
    MolGraph(symbols=('H', 'O', 'H'), bonds=((0, 1), (1, 2)), charge=0, spin=0)
    >>> relabel(relabel(g, [2, 0, 1]), [1, 2, 0]) == g  # [1, 2, 0] is the inverse
    True

    Raises:
        ValueError: If `atom_map` is not a permutation of the atoms.
    """
    n = len(gra.symbols)
    atom_map = [int(i) for i in atom_map]
    if sorted(atom_map) != list(range(n)):
        msg = f"The atom map {atom_map} is not a permutation of 0 to {n - 1}"
        raise ValueError(msg)
    new = [0] * n  # new[old atom] = its index in the result
    for k, i in enumerate(atom_map):
        new[i] = k

    def pairs(bonds: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(new[i], new[j]) for i, j in bonds]

    data: dict[str, Any] = _fields(gra)
    data["symbols"] = [gra.symbols[i] for i in atom_map]
    data["bonds"] = pairs(gra.bonds)
    if isinstance(gra, ReactionGraph | StereoReactionGraph):
        data["breaking"] = pairs(gra.breaking)
        data["forming"] = pairs(gra.forming)
    if isinstance(gra, StereoMolGraph | StereoReactionGraph):
        data["atom_stereo"] = [_relabel_ls(ls, new) for ls in gra.atom_stereo]
        data["bond_stereo"] = [_relabel_ls(ls, new) for ls in gra.bond_stereo]
    if isinstance(gra, StereoReactionGraph):
        data["stereo_changes"] = [
            StereoChange(
                reactant=_relabel_slot(c.reactant, new),
                product=_relabel_slot(c.product, new),
                ts=_relabel_slot(c.ts, new),
            )
            for c in gra.stereo_changes
        ]
    return type(gra)(**data)


@overload
def without_stereo(gra: StereoMolGraph) -> MolGraph: ...
@overload
def without_stereo(gra: StereoReactionGraph) -> ReactionGraph: ...
def without_stereo(
    gra: StereoMolGraph | StereoReactionGraph,
) -> MolGraph | ReactionGraph:
    """The graph without its LS and changes, as a `MolGraph` or a `ReactionGraph`.

    >>> from protomol import Tetrahedral
    >>> g = StereoMolGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    ...     atom_stereo=[Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)],
    ... )
    >>> type(without_stereo(g)).__name__
    'MolGraph'

    Raises:
        TypeError: If the graph has no stereo fields (a `MolGraph` or a
            `ReactionGraph`).
    """
    if not isinstance(gra, StereoMolGraph | StereoReactionGraph):
        msg = f"Expected a StereoMolGraph or StereoReactionGraph, got {gra!r}"
        raise TypeError(msg)
    data = {k: v for k, v in _fields(gra).items() if k not in _STEREO_FIELDS}
    if isinstance(gra, StereoReactionGraph):
        return ReactionGraph(**data)
    return MolGraph(**data)


# Helpers


def _side(
    gra: ReactionGraph | StereoReactionGraph, side: str
) -> MolGraph | StereoMolGraph:
    """The reactant or product side of a reaction graph."""
    dropped = set(gra.forming if side == "reactant" else gra.breaking)
    data: dict[str, Any] = {
        "symbols": gra.symbols,
        "bonds": [bond for bond in gra.bonds if bond not in dropped],
        "charge": gra.charge,
        "spin": gra.spin,
    }
    if isinstance(gra, ReactionGraph):
        return MolGraph(**data)
    lss = [*gra.atom_stereo, *gra.bond_stereo]
    lss.extend(ls for c in gra.stereo_changes if (ls := getattr(c, side)) is not None)
    return StereoMolGraph(**data, **_stereo_fields(lss, reaction=False))


def _relabel_slot(ls: _LS | None, new: Sequence[int]) -> _LS | None:
    return None if ls is None else _relabel_ls(ls, new)


def _relabel_ls[S: _LS](ls: S, new: Sequence[int]) -> S:
    """The LS with each atom i replaced by `new[i]` (None kept)."""
    data = ls.model_dump()
    data["atoms"] = tuple(None if i is None else new[i] for i in ls.atoms)
    if isinstance(ls, PlanarBond | AtropBond):
        data["chain"] = tuple(new[i] for i in ls.chain)
    return type(ls).model_validate(data)
