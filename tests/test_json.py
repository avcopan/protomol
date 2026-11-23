"""Tests for the JSON round trip."""

import json
from typing import Any

import pytest
from pydantic import BaseModel

import protomol as pm
from protomol import AnyGraph

# Step 4 adds from_smiles; remove this mark then
needs_smiles = pytest.mark.skipif(
    not hasattr(pm, "from_smiles"), reason="from_smiles comes in Step 4"
)

CHIRAL: dict[str, Any] = {  # CHFClBr
    "symbols": ["C", "F", "Cl", "Br", "H"],
    "bonds": [(0, 1), (0, 2), (0, 3), (0, 4)],
}
SN2: dict[str, Any] = {  # I⁻ + CHFClBr → CHFClI + Br⁻: C0, F1, Cl2, Br3, H4, I5
    "symbols": ["C", "F", "Cl", "Br", "H", "I"],
    "bonds": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
    "breaking": [(0, 3)],
    "forming": [(0, 5)],
    "charge": -1,
}
TET = pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1)


def _stereo_sn2() -> pm.StereoReactionGraph:
    """The SN2 with Walden inversion at C0, and an undetermined TS slot."""
    change = pm.StereoChange(
        reactant=TET,
        product=pm.Tetrahedral(atoms=(0, 1, 2, 5, 4), parity=1),
        ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
    )
    return pm.StereoReactionGraph(**SN2, stereo_changes=[change])


ALL_TYPES = [
    pytest.param(pm.MolGraph(**CHIRAL), id="MolGraph"),
    pytest.param(pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET]), id="StereoMolGraph"),
    pytest.param(pm.ReactionGraph(**SN2), id="ReactionGraph"),
    pytest.param(_stereo_sn2(), id="StereoReactionGraph"),
]


@needs_smiles
@pytest.mark.parametrize("smiles", ["C", "CO", "C[C@@H](O)/C=C/C", "[CH2]", "[NH4+]"])
def test_round_trip(smiles: str) -> None:
    """from_json(to_json(g)) == g, and to_json is stable."""
    g = pm.from_smiles(smiles)  # ty: ignore[missing-attribute]  Step 4
    string = pm.to_json(g)
    assert pm.from_json(string) == g
    assert pm.to_json(pm.from_json(string)) == string


@pytest.mark.parametrize("g", ALL_TYPES)
def test_every_type_round_trips(g: AnyGraph) -> None:
    """Type, bonds, bond changes, LS and changes survive; the bytes are stable."""
    string = pm.to_json(g)
    copy = pm.from_json(string)
    assert type(copy) is type(g)
    assert copy == g
    assert pm.to_json(copy) == string


def test_bytes_independent_of_insertion_order() -> None:
    """The same graph built from inputs in two orders gives the same bytes."""
    # trans-CHF=CHF: C0=C1, F2 and H3 on C0, F4 and H5 on C1; a made-up lone-pair LS
    # on C0 to have two kinds of LS
    g1 = pm.StereoMolGraph(
        symbols=list("CCFHFH"),
        bonds=[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
        atom_stereo=[pm.Tetrahedral(atoms=(0, 1, 2, 3, None))],
        bond_stereo=[pm.PlanarBond(atoms=(2, 3, 0, 1, 5, 4), parity=0)],
    )
    g2 = pm.StereoMolGraph(
        symbols=list("CCFHFH"),
        bonds=[(5, 1), (4, 1), (3, 0), (2, 0), (1, 0)],
        atom_stereo=[pm.Tetrahedral(atoms=(0, None, 3, 1, 2))],
        bond_stereo=[pm.PlanarBond(atoms=(4, 5, 1, 0, 3, 2), parity=0)],
    )
    assert pm.to_json(g1) == pm.to_json(g2)
    assert g1 == g2


def test_json_form() -> None:
    """Compact; `type`, then the fields in field order; every LS field written."""
    g = _stereo_sn2()
    assert json.loads(pm.to_json(g)) == {
        "type": "StereoReactionGraph",
        "symbols": ["C", "F", "Cl", "Br", "H", "I"],
        "bonds": [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
        "charge": -1,
        "spin": 0,
        "breaking": [[0, 3]],
        "forming": [[0, 5]],
        "atom_stereo": [],
        "bond_stereo": [],
        "stereo_changes": [
            {
                "reactant": {
                    "kind": "Tetrahedral",
                    "atoms": [0, 1, 2, 3, 4],
                    "parity": 1,
                },
                "product": {
                    "kind": "Tetrahedral",
                    "atoms": [0, 1, 2, 4, 5],
                    "parity": -1,
                },
                "ts": {
                    "kind": "Tetrahedral",
                    "atoms": [0, None, 1, 2, 4],
                    "parity": None,
                },
            }
        ],
    }
    assert " " not in pm.to_json(g)
    assert list(json.loads(pm.to_json(g))) == list(pm.StereoReactionGraph.model_fields)
    assert list(json.loads(pm.to_json(pm.ReactionGraph(**SN2)))) == [
        "type",
        "symbols",
        "bonds",
        "charge",
        "spin",
        "breaking",
        "forming",
    ]
    mol = pm.StereoMolGraph(
        symbols=["C", "C"],
        bonds=[(0, 1)],
        bond_stereo=[pm.PlanarBond(atoms=(None, None, 0, 1, None, None))],
    )
    assert json.loads(pm.to_json(mol))["bond_stereo"] == [
        {
            "kind": "PlanarBond",
            "atoms": [None, None, 0, 1, None, None],
            "parity": None,
            "chain": [],
        }
    ]


MOL = '"symbols":["H","H"],"bonds":[[0,1]],"charge":0,"spin":0'


@pytest.mark.parametrize(
    "string",
    [
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param("{" + MOL + "}", id="no-type"),
        pytest.param('{"type":"Graph",' + MOL + "}", id="unknown-type"),
        pytest.param('{"type":["MolGraph"],' + MOL + "}", id="type-not-a-string"),
        pytest.param(
            '{"type":"MolGraph",' + MOL + ',"breaking":[]}', id="field-of-another-type"
        ),
        pytest.param('{"type":"MolGraph",' + MOL + ',"x":1}', id="unknown-field"),
        pytest.param(
            '{"type":"MolGraph","symbols":["H","H"],"bonds":[[0,1],[1,0]]}',
            id="bond-twice",
        ),
        pytest.param(
            '{"type":"ReactionGraph","symbols":["H","H"],"bonds":[],'
            '"breaking":[[0,1]]}',
            id="breaking-not-in-bonds",
        ),
        pytest.param(
            '{"type":"MolGraph","symbols":["H","H"],"bonds":[[0,2]]}',
            id="atom-out-of-range",
        ),
        pytest.param(
            '{"type":"MolGraph","symbols":["H","H"],"bonds":[],"spin":1}',
            id="invalid-graph",
        ),
        pytest.param(
            '{"type":"MolGraph","symbols":["H"],"bonds":[],"charge":"1","spin":0}',
            id="charge-a-string",
        ),
        pytest.param(
            '{"type":"MolGraph","symbols":["H"],"bonds":[],"charge":0,"spin":1.0}',
            id="spin-a-float",
        ),
    ],
)
def test_bad_input_raises(string: str) -> None:
    """Malformed, foreign or invalid input raises GraphError."""
    with pytest.raises(pm.GraphError):
        pm.from_json(string)


def test_two_ls_on_one_site_raises() -> None:
    """The JSON lists each LS site once."""
    data = json.loads(pm.to_json(pm.StereoMolGraph(**CHIRAL, atom_stereo=[TET])))
    data["atom_stereo"] *= 2
    with pytest.raises(pm.GraphError, match="Two LS"):
        pm.from_json(json.dumps(data))


def test_two_changes_on_one_site_raises() -> None:
    """The JSON lists each changing site once."""
    data = json.loads(pm.to_json(_stereo_sn2()))
    data["stereo_changes"] *= 2
    with pytest.raises(pm.GraphError, match="Two changes"):
        pm.from_json(json.dumps(data))


class _Record(BaseModel):
    """A pydantic model with a graph field, as AutoStorage would have."""

    name: str
    graph: AnyGraph


@pytest.mark.parametrize("g", ALL_TYPES)
def test_graph_field_of_a_pydantic_model(g: AnyGraph) -> None:
    """`AnyGraph` is a discriminated union: a model with a graph field round-trips."""
    record = _Record(name="x", graph=g)
    copy = _Record.model_validate_json(record.model_dump_json())
    assert type(copy.graph) is type(g)
    assert copy == record
