"""A deterministic JSON form of every type (P17, P31).

`to_json` is pydantic's compact `model_dump_json`: the fields in field order, `type`
first. The validators have sorted the bonds, LS and changes, so the bytes don't depend
on the order of the input; every field of an LS or a change is written (null and []
included). `from_json` validates with a `TypeAdapter` of the discriminated union
`AnyGraph` in strict mode, so `type` picks the class, which rejects unknown fields and
the fields of another type, and a string or float is not an int.
"""

from pydantic import TypeAdapter, ValidationError

from ._types import AnyGraph, GraphError

_ANY_GRAPH: TypeAdapter[AnyGraph] = TypeAdapter(AnyGraph)


def to_json(g: AnyGraph) -> str:
    """The graph as compact JSON: its fields in field order, `type` first.

    >>> from protomol import MolGraph
    >>> to_json(MolGraph(symbols=["H", "H"], bonds=[(1, 0)]))
    '{"type":"MolGraph","symbols":["H","H"],"bonds":[[0,1]],"charge":0,"spin":0}'
    """
    return g.model_dump_json()


def from_json(string: str) -> AnyGraph:
    """The graph of the type named in the JSON, validated in strict mode.

    >>> from_json('{"type":"MolGraph","symbols":["H","H"],"bonds":[[0,1]],'
    ...           '"charge":0,"spin":0}').bonds
    ((0, 1),)

    Raises:
        GraphError: If the string is not a JSON object, names no known type, has a
            field that doesn't belong to its type, has a value of the wrong JSON type
            (such as `"1"` or `1.0` for an int), or gives an invalid graph.
    """
    try:
        return _ANY_GRAPH.validate_json(string, strict=True)
    except ValidationError as err:
        raise GraphError(str(err)) from err
