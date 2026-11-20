"""The four graph types. Atom i of a graph is atom i of its geometry (P4).

The types are separate classes over one engine, not subclasses of one another (P20).
`charge`, `spin` and the LS maps live in the graph-attribute dict (`self.graph`), so
networkx operations carry them.
"""

import itertools
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

import networkx as nx
from rdkit import Chem

from ._graph import Edge, Graph, Node
from ._stereo import (
    AtomStereo,
    AtropBond,
    BondStereo,
    PlanarBond,
    StereoChange,
    Tetrahedral,
    site,
)
from ._stereo import is_determined as _ls_is_determined

type _LS = Tetrahedral | PlanarBond | AtropBond

_PERIODIC_TABLE = Chem.GetPeriodicTable()
_ATOMIC_NUMBER = {_PERIODIC_TABLE.GetElementSymbol(z): z for z in range(1, 119)}


class GraphError(ValueError):
    """A structurally invalid graph (stereomol's GraphError)."""


class Atom(Node):
    """An atom: its element only (every atom is explicit, H included)."""

    symbol: str


class Bond(Edge):
    """A bond."""


class ReactionBond(Bond):
    """A bond of a reaction or TS graph, with its change."""

    change: Literal["breaking", "forming"] | None = None


class _Base[NodeT: Atom, EdgeT: Bond](Graph[NodeT, EdgeT]):
    """What every type shares: charge, spin and labelled equality."""

    _MAPS: ClassVar[tuple[str, ...]] = ()  # the graph attributes that hold LS maps

    def __init__(self, *, charge: int = 0, spin: int = 0) -> None:
        """A graph with no atoms. Without a spin, the spin is 0 (P27)."""
        super().__init__()
        self.graph["charge"] = charge
        self.graph["spin"] = spin
        for key in self._MAPS:
            self.graph[key] = {}

    @property
    def charge(self) -> int:
        """The total charge."""
        return self.graph["charge"]

    @property
    def spin(self) -> int:
        """Unpaired electrons: 2S = multiplicity - 1 (stereomol D21)."""
        return self.graph["spin"]

    def copy(self, as_view: bool = False) -> Self:  # noqa: FBT001, FBT002  nx's signature
        """A copy with its own LS maps (nx copies the graph attributes shallowly)."""
        gra = super().copy(as_view=as_view)
        if not as_view:
            for key in self._MAPS:
                gra.graph[key] = dict(self.graph[key])
        return gra

    def validate(self) -> None:
        """Check the graph's structure, raising `GraphError`.

        The nodes are `range(n)`, the symbols are elements, the spin is non-negative
        and has the parity of the electron count, and each LS lists the neighbours of
        its atoms.
        """
        try:
            super().validate()
        except ValueError as err:
            raise GraphError(str(err)) from err
        if set(self.nodes) != set(range(self.number_of_nodes())):
            msg = f"The atoms must be 0 to n - 1, got {sorted(self.nodes)}"
            raise GraphError(msg)
        symbols = dict(self.nodes(data="symbol"))
        unknown = {
            symbol for symbol in symbols.values() if symbol not in _ATOMIC_NUMBER
        }
        if unknown:
            msg = f"Unknown element symbols: {sorted(unknown)}"
            raise GraphError(msg)
        if self.spin < 0:
            msg = f"The spin must be non-negative, got {self.spin}"
            raise GraphError(msg)
        electrons = sum(_ATOMIC_NUMBER[symbol] for symbol in symbols.values())
        electrons -= self.charge
        if (electrons - self.spin) % 2:
            msg = f"Spin {self.spin} has the wrong parity for {electrons} electrons"
            raise GraphError(msg)
        self._validate_stereo()

    def _validate_stereo(self) -> None:
        """Check the LS against the bonds (the stereo types override this)."""

    def __eq__(self, other: object) -> bool:
        """Labelled equality: the same type, atoms, bonds, charge, spin and LS (P21)."""
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, _Base)  # for the type checker  # noqa: S101
        return self._key() == other._key()

    __hash__ = None  # mutable

    def _key(self) -> tuple[object, ...]:
        """What `==` compares: what `to_json` will write (Step 3)."""
        nodes = sorted(
            (atom, dict(self.node_type.model_validate(data)))
            for atom, data in self.nodes(data=True)
        )
        edges = sorted(
            (
                min(i, j),
                max(i, j),
                tuple(dict(self.edge_type.model_validate(data)).items()),
            )
            for i, j, data in self.edges(data=True)
        )
        return (nodes, edges, self.graph)


class MolGraph(_Base[Atom, Bond]):
    """A molecular graph."""

    node_type = Atom
    edge_type = Bond


class _StereoMaps:
    """The atom and bond LS maps of the stereo types (keys from `site`)."""

    _MAPS: ClassVar[tuple[str, ...]] = ("atom_stereo", "bond_stereo")
    graph: dict[str, Any]

    @property
    def atom_stereo(self) -> Mapping[int, AtomStereo]:
        """The atom LS, by centre atom."""
        return MappingProxyType(self.graph["atom_stereo"])

    @property
    def bond_stereo(self) -> Mapping[tuple[int, int], BondStereo]:
        """The bond LS, by sorted pair of end atoms."""
        return MappingProxyType(self.graph["bond_stereo"])

    def set_stereo(self, ls: AtomStereo | BondStereo) -> None:
        """Set the LS of a site, replacing any LS there."""
        self.graph[_map_key(ls)][site(ls)] = ls

    def remove_stereo(self, key: int | tuple[int, int]) -> None:
        """Remove the LS of a site, from whichever map holds it.

        Raises:
            KeyError: If no map holds the site.
        """
        maps = [self.graph[name] for name in self._MAPS]
        holders = [stereo for stereo in maps if key in stereo]
        if not holders:
            raise KeyError(key)
        for stereo in holders:
            del stereo[key]


class StereoMolGraph(_StereoMaps, _Base[Atom, Bond]):
    """A molecular graph with local stereodescriptors (LS)."""

    node_type = Atom
    edge_type = Bond

    def _validate_stereo(self) -> None:
        for key, ls in _items(self, self._MAPS):
            _check_ls(key, ls, self, exact=True)


class ReactionGraph(_Base[Atom, ReactionBond]):
    """A reaction or TS graph: bonds that break or form, and bonds that don't change."""

    node_type = Atom
    edge_type = ReactionBond


class StereoReactionGraph(_StereoMaps, _Base[Atom, ReactionBond]):
    """A reaction or TS graph with stereo.

    `atom_stereo` and `bond_stereo` hold the LS that don't change. The changing ones are
    in the change maps; a site is in one or the other, never both.
    """

    _MAPS: ClassVar[tuple[str, ...]] = (
        "atom_stereo",
        "bond_stereo",
        "atom_stereo_changes",
        "bond_stereo_changes",
    )
    node_type = Atom
    edge_type = ReactionBond

    @property
    def atom_stereo_changes(self) -> Mapping[int, StereoChange]:
        """The changing atom sites, by centre atom."""
        return MappingProxyType(self.graph["atom_stereo_changes"])

    @property
    def bond_stereo_changes(self) -> Mapping[tuple[int, int], StereoChange]:
        """The changing bond sites, by sorted pair of end atoms."""
        return MappingProxyType(self.graph["bond_stereo_changes"])

    def set_stereo_change(self, change: StereoChange) -> None:
        """Set the change of a site, replacing any change there."""
        ls = next(ls for ls in (change.reactant, change.product, change.ts) if ls)
        self.graph[_map_key(ls) + "_changes"][site(ls)] = change

    def _validate_stereo(self) -> None:
        both = (set(self.atom_stereo) & set(self.atom_stereo_changes)) | (
            set(self.bond_stereo) & set(self.bond_stereo_changes)
        )
        if both:
            msg = f"Sites both unchanged and changing: {sorted(both, key=str)}"
            raise GraphError(msg)
        reactant = _side(self, "forming")
        product = _side(self, "breaking")
        # An unchanged LS holds on both sides
        for key, ls in _items(self, ("atom_stereo", "bond_stereo")):
            _check_ls(key, ls, reactant, exact=True)
            _check_ls(key, ls, product, exact=True)
        # A changing LS holds on its side; the TS slot only needs bonded atoms (a
        # reacting centre can have five neighbours in the TS graph)
        for key, change in _items(self, ("atom_stereo_changes", "bond_stereo_changes")):
            if change.reactant:
                _check_ls(key, change.reactant, reactant, exact=True)
            if change.product:
                _check_ls(key, change.product, product, exact=True)
            if change.ts:
                _check_ls(key, change.ts, self, exact=False)


def is_determined(gra: StereoMolGraph | StereoReactionGraph) -> bool:
    """Whether every LS is determined (a workflow graph needs this).

    The `ts` slot of a change may stay undetermined (post-TS bifurcation, P5).
    """
    lss: list[_LS] = [*gra.atom_stereo.values(), *gra.bond_stereo.values()]
    if isinstance(gra, StereoReactionGraph):
        changes = [*gra.atom_stereo_changes.values(), *gra.bond_stereo_changes.values()]
        lss.extend(
            ls for change in changes for ls in (change.reactant, change.product) if ls
        )
    return all(map(_ls_is_determined, lss))


def _map_key(ls: _LS) -> str:
    return "atom_stereo" if isinstance(ls, Tetrahedral) else "bond_stereo"


def _items(gra: _Base, names: Collection[str]) -> list[tuple[Any, Any]]:
    return [item for name in names for item in gra.graph[name].items()]


def _side(gra: ReactionGraph | StereoReactionGraph, absent: str) -> nx.Graph:
    """One side of a reaction graph: its bonds without the `absent` change."""
    return nx.restricted_view(
        gra,
        (),
        [(i, j) for i, j, change in gra.edges(data="change") if change == absent],
    )


def _check_ls(key: object, ls: _LS, gra: nx.Graph, *, exact: bool) -> None:
    """Check that `ls`, keyed by `key`, lists the neighbours of its atoms in `gra`.

    With `exact`, the listed atoms are exactly the neighbours. Without it, each is a
    neighbour (the TS slot).
    """
    if key != site(ls):
        msg = f"LS {ls} is keyed by {key}, not by its site {site(ls)}"
        raise GraphError(msg)
    missing = {atom for atom in (*ls.atoms, *_chain(ls)) if atom is not None} - set(gra)
    if missing:
        msg = f"LS {ls} has atoms {sorted(missing)} that are not in the graph"
        raise GraphError(msg)
    if isinstance(ls, Tetrahedral):
        center, *neighbors = ls.atoms
        _check_neighbors(ls, center, neighbors, set(gra[center]), exact=exact)
        return
    a, b, end1, end2, c, d = ls.atoms
    path = [end1, *ls.chain, end2]
    for i, j in itertools.pairwise(path):
        if not gra.has_edge(i, j):
            msg = f"LS {ls}: {path} is not a path of bonds"
            raise GraphError(msg)
    if exact:
        bad = [atom for atom in ls.chain if gra.degree(atom) != 2]  # noqa: PLR2004
        if bad:
            msg = f"LS {ls}: inner chain atoms {bad} don't have two neighbours"
            raise GraphError(msg)
    _check_neighbors(ls, end1, (a, b), set(gra[end1]) - {path[1]}, exact=exact)
    _check_neighbors(ls, end2, (c, d), set(gra[end2]) - {path[-2]}, exact=exact)


def _check_neighbors(
    ls: _LS,
    atom: int,
    listed: Collection[int | None],
    neighbors: Collection[int],
    *,
    exact: bool,
) -> None:
    """Check the atoms that `ls` lists for `atom` against its `neighbors`."""
    listed = sorted(n for n in listed if n is not None)
    neighbors = sorted(neighbors)
    ok = (
        listed == neighbors
        if exact
        else len(set(listed)) == len(listed) and set(listed) <= set(neighbors)
    )
    if not ok:
        rule = "exactly" if exact else "only"
        msg = (
            f"LS {ls} lists {listed} on atom {atom}, which must be {rule} its "
            f"neighbours {neighbors}"
        )
        raise GraphError(msg)


def _chain(ls: _LS) -> tuple[int, ...]:
    return () if isinstance(ls, Tetrahedral) else ls.chain
