"""A deterministic JSON form of every type (P17).

The bytes don't depend on insertion order: bonds are sorted pairs `i < j` in sorted
order, LS (normalized on construction) are sorted by site, and changes are sorted with
the atom sites first, then the bond sites. Each type writes only its own fields, and
every field of an LS or a change (null and [] included).
"""

from typing import Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, ValidationError

from ._stereo import AtomStereo, BondStereo, StereoChange, site
from ._types import (
    GraphError,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
)

type AnyGraph = MolGraph | StereoMolGraph | ReactionGraph | StereoReactionGraph
type _Pair = tuple[int, int]

_TYPES: dict[str, type[AnyGraph]] = {
    cls.__name__: cls
    for cls in (MolGraph, StereoMolGraph, ReactionGraph, StereoReactionGraph)
}
_REACTION_TYPES = (ReactionGraph, StereoReactionGraph)
_REACTION_FIELDS = {"breaking", "forming"}
_STEREO_FIELDS = {"atom_stereo", "bond_stereo"}
_FIELDS: dict[str, set[str]] = {
    "MolGraph": {"type", "charge", "spin", "symbols", "bonds"},
}
_FIELDS["StereoMolGraph"] = _FIELDS["MolGraph"] | _STEREO_FIELDS
_FIELDS["ReactionGraph"] = _FIELDS["MolGraph"] | _REACTION_FIELDS
_FIELDS["StereoReactionGraph"] = (
    _FIELDS["MolGraph"] | _REACTION_FIELDS | _STEREO_FIELDS | {"stereo_changes"}
)


class _GraphJSON(BaseModel):
    """The schema. The field order is the order in the JSON."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["MolGraph", "StereoMolGraph", "ReactionGraph", "StereoReactionGraph"]
    charge: int
    spin: int
    symbols: list[str]
    bonds: list[_Pair]  # unchanged bonds, i < j
    breaking: list[_Pair] = []  # reaction types only
    forming: list[_Pair] = []
    atom_stereo: list[AtomStereo] = []  # stereo types only
    bond_stereo: list[BondStereo] = []
    stereo_changes: list[StereoChange] = []  # StereoReactionGraph only


def to_json(gra: AnyGraph) -> str:
    """The graph as compact JSON with sorted bonds and normalized LS.

    >>> gra = MolGraph()
    >>> gra.add_nodes_from([(1, {"symbol": "H"}), (0, {"symbol": "H"})])
    >>> gra.add_edge(1, 0)
    >>> to_json(gra)
    '{"type":"MolGraph","charge":0,"spin":0,"symbols":["H","H"],"bonds":[[0,1]]}'

    Raises:
        GraphError: If the atoms are not 0 to n - 1.
    """
    name = type(gra).__name__
    n = gra.number_of_nodes()
    if set(gra.nodes) != set(range(n)):
        msg = f"The atoms must be 0 to n - 1, got {sorted(gra.nodes)}"
        raise GraphError(msg)
    bonds: dict[str | None, list[_Pair]] = {None: [], "breaking": [], "forming": []}
    for i, j, change in gra.edges(data="change"):
        bonds[change if isinstance(gra, _REACTION_TYPES) else None].append(
            (min(i, j), max(i, j))
        )
    data = _GraphJSON(
        type=name,  # ty: ignore[invalid-argument-type]  one of the four names
        charge=gra.charge,
        spin=gra.spin,
        symbols=[gra.nodes[i]["symbol"] for i in range(n)],
        bonds=sorted(bonds[None]),
        breaking=sorted(bonds["breaking"]),
        forming=sorted(bonds["forming"]),
    )
    if isinstance(gra, StereoMolGraph | StereoReactionGraph):
        data.atom_stereo = [gra.atom_stereo[key] for key in sorted(gra.atom_stereo)]
        data.bond_stereo = [gra.bond_stereo[key] for key in sorted(gra.bond_stereo)]
    if isinstance(gra, StereoReactionGraph):
        data.stereo_changes = [
            *(gra.atom_stereo_changes[k] for k in sorted(gra.atom_stereo_changes)),
            *(gra.bond_stereo_changes[k] for k in sorted(gra.bond_stereo_changes)),
        ]
    return data.model_dump_json(include=_FIELDS[name])


def from_json(string: str) -> AnyGraph:
    """The graph of the type named in the JSON, validated.

    >>> from_json('{"type":"MolGraph","charge":0,"spin":0,"symbols":["H","H"],'
    ...           '"bonds":[[0,1]]}').number_of_edges()
    1

    Raises:
        GraphError: If the string is not JSON in this schema, has a field that doesn't
            belong to its type, lists a bond or an LS site twice, or gives an invalid
            graph.
    """
    try:
        data = _GraphJSON.model_validate_json(string)
    except ValidationError as err:
        raise GraphError(str(err)) from err
    extra = data.model_fields_set - _FIELDS[data.type]
    if extra:
        msg = f"A {data.type} has no fields {sorted(extra)}"
        raise GraphError(msg)

    gra = _TYPES[data.type](charge=data.charge, spin=data.spin)
    gra.add_nodes_from((i, {"symbol": symbol}) for i, symbol in enumerate(data.symbols))
    _add_bonds(gra, data.bonds)
    _add_bonds(gra, data.breaking, change="breaking")
    _add_bonds(gra, data.forming, change="forming")
    if isinstance(gra, StereoMolGraph | StereoReactionGraph):
        for ls in [*data.atom_stereo, *data.bond_stereo]:
            if site(ls) in gra.atom_stereo or site(ls) in gra.bond_stereo:
                msg = f"Two LS on site {site(ls)}"
                raise GraphError(msg)
            gra.set_stereo(ls)
    if isinstance(gra, StereoReactionGraph):
        for change in data.stereo_changes:
            key = _change_site(change)
            if key in gra.atom_stereo_changes or key in gra.bond_stereo_changes:
                msg = f"Two changes on site {key}"
                raise GraphError(msg)
            gra.set_stereo_change(change)
    gra.validate()
    return gra


def _add_bonds(gra: nx.Graph, bonds: list[_Pair], **attrs: str) -> None:
    for i, j in bonds:
        if gra.has_edge(i, j):
            msg = f"Bond {(i, j)} is listed twice"
            raise GraphError(msg)
        gra.add_edge(i, j, **attrs)


def _change_site(change: StereoChange) -> int | tuple[int, int]:
    ls = next(ls for ls in (change.reactant, change.product, change.ts) if ls)
    return site(ls)
