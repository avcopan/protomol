"""The four graph types: frozen pydantic models whose fields are the JSON fields (P34).

Atom i of a graph is atom i of its geometry (P4). The types are separate classes, not
subclasses of one another (P20); a private base holds their shared fields (`type`,
`symbols`, `bonds`, `charge`, `spin`, in that order, before each type's own fields),
configuration and checks. The fields are the constructor's keyword arguments and the
JSON's fields, in one order. `type` is the class name, so `AnyGraph` is a discriminated
union that a field of another pydantic model can use (P31). Validation is pydantic's
lax mode, so a numpy array of bonds or an `np.int64` atom becomes an `int`, while a
float with a fractional part fails (`from_json` is strict).

Per-field checks are annotated types (`_Symbols`, `_Pairs`, ...) that normalize (bonds
as sorted pairs `i < j`, LS and changes sorted by site), so `==` and `hash` are field
equality (P21). The checks that span fields, the LS rules of P30 among them, are one
model validator, so a graph that exists is valid (P35).

Errors: the constructors raise pydantic's `ValidationError`, which is a `ValueError`.
Catch `ValueError`: it also covers `GraphError` (raised by `from_json`) and stereomol's
`GraphError`, so it is the portable rule. `with_stereo` raises `TypeError` for an item
of the wrong kind.

Changes are functions that build a new graph through the constructor (P36). Never use
`model_copy(update=...)` or `model_construct` on a graph: they skip validation, so they
can make a graph that is invalid or not normalized.

networkx is private: `_nx(g)` is a cached, frozen networkx graph for the algorithms.
"""

import functools
import itertools
from collections.abc import Collection, Iterable, Mapping
from typing import Annotated, Any, Literal, Self, overload

import networkx as nx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)
from rdkit import Chem

from ._stereo import (
    _ATOM_KINDS,
    AtomStereo,
    AtropBond,
    BondStereo,
    Octahedral,
    PlanarBond,
    SquarePlanar,
    StereoChange,
    Tetrahedral,
    TrigonalBipyramidal,
)
from ._stereo import is_determined as _ls_is_determined
from ._stereo import site as _ls_site

type _LS = (
    Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
)
type _Pair = tuple[int, int]
type _Site = int | _Pair
type _Adjacency = Mapping[int, Collection[int]]
type _StereoGraph = StereoMolGraph | StereoReactionGraph

_PERIODIC_TABLE = Chem.GetPeriodicTable()
_ATOMIC_NUMBER = {_PERIODIC_TABLE.GetElementSymbol(z): z for z in range(1, 119)}


class GraphError(ValueError):
    """Invalid graph input (stereomol's GraphError).

    Raised by `from_json`, and for graphs of mixed types given to `isomorphism` or
    `deduplicate`.
    """


def _lowest_spin(data: dict[str, Any]) -> int:
    """The default spin: the lowest the electron count allows (0 or 1, sm D21).

    `data` holds the fields validated so far. Without valid symbols or charge, 0; their
    error is reported anyway (newer pydantic doesn't call this after an error, but
    2.10 does).
    """
    if "symbols" not in data or "charge" not in data:
        return 0
    electrons = sum(_ATOMIC_NUMBER[symbol] for symbol in data["symbols"])
    return (electrons - data["charge"]) % 2


# Per-field checks (checks that span fields are in `_Graph._check_graph`)


def _check_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(symbols) - _ATOMIC_NUMBER.keys())
    if unknown:
        msg = f"Unknown element symbols: {unknown}"
        raise ValueError(msg)
    return symbols


def _normalize_pairs(bonds: Iterable[_Pair]) -> tuple[_Pair, ...]:
    """Sorted pairs `i < j` in sorted order; no loops, negative atoms or repeats."""
    pairs: set[_Pair] = set()
    for i, j in bonds:
        pair = (min(i, j), max(i, j))
        if i == j or pair[0] < 0 or pair in pairs:
            problem = "joins an atom to itself" if i == j else "is listed twice"
            problem = "has a negative atom" if pair[0] < 0 else problem
            msg = f"The bond {(i, j)} {problem}"
            raise ValueError(msg)
        pairs.add(pair)
    return tuple(sorted(pairs))


def _sorted_ls[T: _LS](lss: tuple[T, ...]) -> tuple[T, ...]:
    return _sorted_by_site(lss, "LS")


def _sorted_changes(changes: tuple[StereoChange, ...]) -> tuple[StereoChange, ...]:
    return _sorted_by_site(changes, "changes")


type _Symbols = Annotated[tuple[str, ...], AfterValidator(_check_symbols)]
type _Pairs = Annotated[tuple[_Pair, ...], AfterValidator(_normalize_pairs)]
type _AtomStereos = Annotated[tuple[AtomStereo, ...], AfterValidator(_sorted_ls)]
type _BondStereos = Annotated[tuple[BondStereo, ...], AfterValidator(_sorted_ls)]
type _Changes = Annotated[tuple[StereoChange, ...], AfterValidator(_sorted_changes)]


class _Graph(BaseModel):
    """The fields, configuration and checks that the four types share (not a type).

    Each type narrows `type` to `Literal["<its name>"]`; it is declared here so that it
    comes first in field order (and so first in the JSON).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(repr=False)
    symbols: _Symbols
    bonds: _Pairs
    charge: int = 0
    spin: NonNegativeInt = Field(default_factory=_lowest_spin)

    @model_validator(mode="after")
    def _check_graph(self) -> Self:
        n = len(self.symbols)
        outside = [bond for bond in self.bonds if bond[1] >= n]
        if outside:
            msg = f"Bonds {outside} have atoms outside 0 to {n - 1}"
            raise ValueError(msg)
        electrons = sum(_ATOMIC_NUMBER[symbol] for symbol in self.symbols) - self.charge
        if (electrons - self.spin) % 2:
            msg = f"Spin {self.spin} has the wrong parity for {electrons} electrons"
            raise ValueError(msg)
        if isinstance(self, ReactionGraph | StereoReactionGraph):
            _check_changing_bonds(self)
        if isinstance(self, StereoMolGraph | StereoReactionGraph):
            _check_stereo(self)
        return self

    def _repr_svg_(self) -> str:
        """The drawing Jupyter shows for the graph (see `protomol.display`)."""
        from ._display import _svg  # noqa: PLC0415  (_display imports this module)

        return _svg(self)  # ty: ignore[invalid-argument-type]  (_Graph is one of the four)


class MolGraph(_Graph):
    """A molecular graph: element symbols and bonds (pairs of atom indices).

    >>> MolGraph(symbols=["O", "H", "H"], bonds=[(1, 0), (0, 2)])
    MolGraph(symbols=('O', 'H', 'H'), bonds=((0, 1), (0, 2)), charge=0, spin=0)

    The spin (unpaired electrons, 2S) defaults to the lowest the electron count allows:

    >>> MolGraph(symbols=["C", "H", "H", "H"], bonds=[(0, 1), (0, 2), (0, 3)]).spin
    1
    """

    type: Literal["MolGraph"] = Field(default="MolGraph", repr=False)


class StereoMolGraph(_Graph):
    """A molecular graph with local stereodescriptors (LS).

    `atom_stereo` holds the atom LS (sorted by centre) and `bond_stereo` the bond LS
    (sorted by end pair). Each LS lists exactly the neighbours of its atoms (P30).

    >>> g = StereoMolGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    ...     atom_stereo=[Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)],
    ... )
    >>> g.atom_stereo[0].atoms
    (0, 1, 2, 3, 4)
    """

    type: Literal["StereoMolGraph"] = Field(default="StereoMolGraph", repr=False)
    atom_stereo: _AtomStereos = ()
    bond_stereo: _BondStereos = ()


class ReactionGraph(_Graph):
    """A reaction or TS graph.

    `bonds` is every bond (the condensed graph's edges), and `breaking` and `forming`
    are disjoint subsets of it: the reactant side is `bonds` without `forming`, the
    product side `bonds` without `breaking`.

    >>> g = ReactionGraph(
    ...     symbols=["H", "H", "H"],
    ...     bonds=[(0, 1), (1, 2)],
    ...     breaking=[(0, 1)],
    ...     forming=[(2, 1)],
    ... )
    >>> g.forming
    ((1, 2),)
    """

    type: Literal["ReactionGraph"] = Field(default="ReactionGraph", repr=False)
    breaking: _Pairs = ()
    forming: _Pairs = ()


class StereoReactionGraph(_Graph):
    """A reaction or TS graph with stereo.

    `atom_stereo` and `bond_stereo` hold the LS that don't change; each holds on both
    sides. `stereo_changes` holds the sites that change (atom sites by centre, then
    bond sites by end pair); a site is in one or the other, never both (P30).

    >>> tet = Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)
    >>> g = StereoReactionGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H", "I"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    ...     breaking=[(0, 3)],
    ...     forming=[(0, 5)],
    ...     charge=-1,
    ...     stereo_changes=[StereoChange(reactant=tet)],
    ... )
    >>> g.stereo_changes[0].reactant == tet
    True
    """

    type: Literal["StereoReactionGraph"] = Field(
        default="StereoReactionGraph", repr=False
    )
    breaking: _Pairs = ()
    forming: _Pairs = ()
    atom_stereo: _AtomStereos = ()
    bond_stereo: _BondStereos = ()
    stereo_changes: _Changes = ()


type AnyGraph = Annotated[
    MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph,
    Field(discriminator="type"),
]
"""Any of the four graph types, discriminated by `type` (for a field of a pydantic
model, or a `TypeAdapter`)."""


# Functions


@overload
def stereo_at(g: StereoMolGraph, site: _Site) -> AtomStereo | BondStereo | None: ...
@overload
def stereo_at(
    g: StereoReactionGraph, site: _Site
) -> AtomStereo | BondStereo | StereoChange | None: ...
def stereo_at(
    g: _StereoGraph, site: _Site
) -> AtomStereo | BondStereo | StereoChange | None:
    """The LS or change at a site (a centre atom or a bond's ends), or None.

    >>> g = StereoMolGraph(
    ...     symbols=["C", "C", "H", "H", "H", "H"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
    ...     bond_stereo=[PlanarBond(atoms=(2, 3, 0, 1, 4, 5), parity=0)],
    ... )
    >>> stereo_at(g, (1, 0)).parity
    0
    >>> stereo_at(g, 0) is None
    True
    """
    key = _normal_site(site)
    return next((item for item in _stereo_items(g) if _site(item) == key), None)


@overload
def with_stereo(g: MolGraph | StereoMolGraph, *items: _LS) -> StereoMolGraph: ...
@overload
def with_stereo(
    g: ReactionGraph | StereoReactionGraph, *items: _LS | StereoChange
) -> StereoReactionGraph: ...
def with_stereo(
    g: AnyGraph, *items: _LS | StereoChange
) -> StereoMolGraph | StereoReactionGraph:
    """A new graph with each item at its site, replacing what was there.

    An item is an LS or, for a reaction graph, a `StereoChange`; in a reaction graph an
    LS and a change at one site replace each other. A `MolGraph` gives a
    `StereoMolGraph` and a `ReactionGraph` a `StereoReactionGraph`.

    >>> g = MolGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H"], bonds=[(0, 1), (0, 2), (0, 3), (0, 4)]
    ... )
    >>> new = with_stereo(g, Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    >>> type(new).__name__, len(new.atom_stereo)
    ('StereoMolGraph', 1)

    Raises:
        TypeError: If an item is not an LS or a `StereoChange`, or is a
            `StereoChange` for a molecule graph.
        ValueError: If the new graph is invalid (pydantic's `ValidationError`), for
            example with two items at one site.
    """
    reaction = isinstance(g, ReactionGraph | StereoReactionGraph)
    for item in items:
        if not isinstance(item, (*_ATOM_KINDS, PlanarBond, AtropBond, StereoChange)):
            msg = f"Expected an LS or a StereoChange, got {item!r}"
            raise TypeError(msg)
        if isinstance(item, StereoChange) and not reaction:
            msg = f"A {type(g).__name__} takes no StereoChange, got {item!r}"
            raise TypeError(msg)
    sites = {_site(item) for item in items}
    kept = [item for item in _stereo_items(g) if _site(item) not in sites]
    data: dict[str, Any] = {
        **_fields(g),
        **_stereo_fields([*kept, *items], reaction=reaction),
    }
    if reaction:
        return StereoReactionGraph(**data)
    return StereoMolGraph(**data)


def remove_stereo[G: (StereoMolGraph, StereoReactionGraph)](g: G, site: _Site) -> G:
    """A new graph without the LS or change at a site.

    >>> g = StereoMolGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    ...     atom_stereo=[Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)],
    ... )
    >>> remove_stereo(g, 0).atom_stereo
    ()

    Raises:
        KeyError: If there is no stereo at the site.
    """
    key = _normal_site(site)
    items = _stereo_items(g)
    kept = [item for item in items if _site(item) != key]
    if len(kept) == len(items):
        raise KeyError(site)
    reaction = isinstance(g, StereoReactionGraph)
    data: dict[str, Any] = {**_fields(g), **_stereo_fields(kept, reaction=reaction)}
    return type(g)(**data)


def neighbors(g: AnyGraph, i: int) -> tuple[int, ...]:
    """The sorted neighbours of atom `i`, over every bond (breaking and forming too).

    >>> neighbors(MolGraph(symbols=["O", "H", "H"], bonds=[(0, 1), (0, 2)]), 0)
    (1, 2)

    Raises:
        IndexError: If `i` is not an atom of the graph.
    """
    if not 0 <= i < len(g.symbols):
        msg = f"Atom {i} is not in a graph of {len(g.symbols)} atoms"
        raise IndexError(msg)
    return tuple(sorted(_nx(g)[i]))


def is_determined(g: _StereoGraph) -> bool:
    """Whether every LS is determined (a workflow graph needs this).

    The `ts` slot of a change may stay undetermined (post-TS bifurcation, P5).

    >>> g = StereoMolGraph(
    ...     symbols=["C", "F", "Cl", "Br", "H"],
    ...     bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
    ...     atom_stereo=[Tetrahedral(atoms=(0, 1, 2, 3, 4))],
    ... )
    >>> is_determined(g)
    False
    """
    lss: list[_LS] = [*g.atom_stereo, *g.bond_stereo]
    if isinstance(g, StereoReactionGraph):
        lss.extend(
            ls
            for change in g.stereo_changes
            for ls in (change.reactant, change.product)
            if ls is not None
        )
    return all(map(_ls_is_determined, lss))


@functools.lru_cache(maxsize=1024)
def _nx(g: AnyGraph) -> nx.Graph:
    """The graph as a frozen networkx graph, for the algorithms.

    Nodes carry `symbol`; a reaction graph's edges carry `change` ("breaking",
    "forming" or None); the graph carries `charge` and `spin`.
    """
    nxg = nx.Graph(charge=g.charge, spin=g.spin)
    nxg.add_nodes_from((i, {"symbol": symbol}) for i, symbol in enumerate(g.symbols))
    if isinstance(g, ReactionGraph | StereoReactionGraph):
        change = dict.fromkeys(g.breaking, "breaking")
        change |= dict.fromkeys(g.forming, "forming")
        nxg.add_edges_from((*bond, {"change": change.get(bond)}) for bond in g.bonds)
    else:
        nxg.add_edges_from(g.bonds)
    return nx.freeze(nxg)


# Validation


def _check_changing_bonds(g: ReactionGraph | StereoReactionGraph) -> None:
    """Breaking and forming bonds are disjoint subsets of `bonds`."""
    for name, bonds in (("breaking", g.breaking), ("forming", g.forming)):
        missing = sorted(set(bonds) - set(g.bonds))
        if missing:
            msg = f"The {name} bonds {missing} must be in `bonds`"
            raise ValueError(msg)
    both = sorted(set(g.breaking) & set(g.forming))
    if both:
        msg = f"Bonds {both} are both breaking and forming"
        raise ValueError(msg)


def _sorted_by_site[T: _LS | StereoChange](
    items: Iterable[T], what: str
) -> tuple[T, ...]:
    """The items sorted by site (atom sites first), raising if two share a site."""
    by_site: dict[_Site, T] = {}
    for item in items:
        key = _site(item)
        if key in by_site:
            msg = f"Two {what} on site {key}"
            raise ValueError(msg)
        by_site[key] = item
    return tuple(by_site[key] for key in sorted(by_site, key=_site_key))


def _check_stereo(g: _StereoGraph) -> None:
    """Check each LS against the bonds of the side where it lives (P30)."""
    n = len(g.symbols)
    lss = [*g.atom_stereo, *g.bond_stereo]
    if isinstance(g, StereoMolGraph):
        adjacency = _adjacency(n, g.bonds)
        for ls in lss:
            _check_ls(ls, adjacency, exact=True)
        return
    both = {_site(ls) for ls in lss} & {_site(c) for c in g.stereo_changes}
    if both:
        msg = f"Sites both unchanged and changing: {sorted(both, key=_site_key)}"
        raise ValueError(msg)
    reactant = _adjacency(n, set(g.bonds) - set(g.forming))
    product = _adjacency(n, set(g.bonds) - set(g.breaking))
    whole = _adjacency(n, g.bonds)
    # An unchanged LS holds on both sides
    for ls in lss:
        _check_ls(ls, reactant, exact=True)
        _check_ls(ls, product, exact=True)
    # A changing LS holds on its side; the TS slot only needs bonded atoms (a reacting
    # centre can have five neighbours in the TS graph)
    for change in g.stereo_changes:
        if change.reactant:
            _check_ls(change.reactant, reactant, exact=True)
        if change.product:
            _check_ls(change.product, product, exact=True)
        if change.ts:
            _check_ls(change.ts, whole, exact=False)


def _adjacency(n: int, bonds: Iterable[_Pair]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def _check_ls(ls: _LS, adjacency: _Adjacency, *, exact: bool) -> None:
    """Check that `ls` lists the neighbours of its atoms.

    With `exact`, the listed atoms are exactly the neighbours. Without it, each is a
    neighbour (the TS slot).
    """
    atoms = {atom for atom in (*ls.atoms, *_chain(ls)) if atom is not None}
    missing = atoms - set(adjacency)
    if missing:
        msg = f"LS {ls} has atoms {sorted(missing)} that are not in the graph"
        raise ValueError(msg)
    if isinstance(ls, _ATOM_KINDS):
        center, *neighbors_ = ls.atoms
        _check_neighbors(ls, center, neighbors_, adjacency[center], exact=exact)
        return
    a, b, end1, end2, c, d = ls.atoms
    path = [end1, *ls.chain, end2]
    for i, j in itertools.pairwise(path):
        if j not in adjacency[i]:
            msg = f"LS {ls}: {path} is not a path of bonds"
            raise ValueError(msg)
    if exact:
        bad = [atom for atom in ls.chain if len(adjacency[atom]) != 2]  # noqa: PLR2004
        if bad:
            msg = f"LS {ls}: inner chain atoms {bad} don't have two neighbours"
            raise ValueError(msg)
    _check_neighbors(ls, end1, (a, b), set(adjacency[end1]) - {path[1]}, exact=exact)
    _check_neighbors(ls, end2, (c, d), set(adjacency[end2]) - {path[-2]}, exact=exact)


def _check_neighbors(
    ls: _LS,
    atom: int,
    listed: Collection[int | None],
    neighbors_: Collection[int],
    *,
    exact: bool,
) -> None:
    """Check the atoms that `ls` lists for `atom` against its neighbours."""
    listed = sorted(n for n in listed if n is not None)
    neighbors_ = sorted(neighbors_)
    ok = (
        listed == neighbors_
        if exact
        else len(set(listed)) == len(listed) and set(listed) <= set(neighbors_)
    )
    if not ok:
        rule = "exactly" if exact else "only"
        msg = (
            f"LS {ls} lists {listed} on atom {atom}, which must be {rule} its "
            f"neighbours {neighbors_}"
        )
        raise ValueError(msg)


# Helpers


def _chain(ls: _LS) -> tuple[int, ...]:
    return () if isinstance(ls, _ATOM_KINDS) else ls.chain


def _site(item: _LS | StereoChange) -> _Site:
    """The site of an LS, or of a change's LS (they share one)."""
    if isinstance(item, StereoChange):
        item = next(ls for ls in (item.reactant, item.product, item.ts) if ls)
    return _ls_site(item)


def _normal_site(site: _Site) -> _Site:
    if isinstance(site, tuple):
        i, j = site
        return (min(i, j), max(i, j))
    return site


def _site_key(site: _Site) -> tuple[int, ...]:
    """Atom sites by centre, then bond sites by end pair."""
    return (0, site) if isinstance(site, int) else (1, *site)


def _stereo_items(g: AnyGraph) -> tuple[_LS | StereoChange, ...]:
    """Every LS and change of the graph (none for a graph without stereo)."""
    if isinstance(g, MolGraph | ReactionGraph):
        return ()
    items: tuple[_LS | StereoChange, ...] = (*g.atom_stereo, *g.bond_stereo)
    if isinstance(g, StereoReactionGraph):
        items += g.stereo_changes
    return items


def _fields(g: AnyGraph) -> dict[str, Any]:
    """The fields of the graph without `type`, to build a graph of another type."""
    return {name: value for name, value in g if name != "type"}


def _stereo_fields(
    items: Iterable[_LS | StereoChange], *, reaction: bool
) -> dict[str, list[Any]]:
    """The stereo fields that hold `items` (the constructor sorts them)."""
    items = list(items)
    fields: dict[str, list[Any]] = {
        "atom_stereo": [x for x in items if isinstance(x, _ATOM_KINDS)],
        "bond_stereo": [x for x in items if isinstance(x, PlanarBond | AtropBond)],
    }
    if reaction:
        fields["stereo_changes"] = [x for x in items if isinstance(x, StereoChange)]
    return fields
