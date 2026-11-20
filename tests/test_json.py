"""Tests for the JSON round trip."""

import json

import pytest

import protomol as pm

type AnyGraph = (
    pm.MolGraph | pm.StereoMolGraph | pm.ReactionGraph | pm.StereoReactionGraph
)

# Step 4 adds from_smiles; remove this mark then
needs_smiles = pytest.mark.skipif(
    not hasattr(pm, "from_smiles"), reason="from_smiles comes in Step 4"
)


def _build[G: AnyGraph](  # noqa: PLR0913
    cls: type[G],
    symbols: str,
    bonds: list[tuple[int, int]],
    *,
    breaking: tuple[tuple[int, int], ...] = (),
    forming: tuple[tuple[int, int], ...] = (),
    charge: int = 0,
) -> G:
    """A graph with one-letter symbols (`X` for Cl, `Y` for Br, `I` for I)."""
    names = {"X": "Cl", "Y": "Br"}
    gra = cls(charge=charge)
    for atom, symbol in enumerate(symbols):
        gra.add_node(atom, symbol=names.get(symbol, symbol))
    gra.add_edges_from(bonds)
    gra.add_edges_from(breaking, change="breaking")
    gra.add_edges_from(forming, change="forming")
    return gra


def _sn2[G: pm.ReactionGraph | pm.StereoReactionGraph](cls: type[G]) -> G:
    """I⁻ + CHFClBr → CHFClI + Br⁻: C0, F1, Cl2, Br3, H4, I5."""
    return _build(
        cls,
        "CFXYHI",
        [(0, 1), (0, 2), (0, 4)],
        breaking=((0, 3),),
        forming=((0, 5),),
        charge=-1,
    )


def _stereo_sn2() -> pm.StereoReactionGraph:
    """The SN2 with Walden inversion at C0, and an undetermined TS slot."""
    gra = _sn2(pm.StereoReactionGraph)
    gra.set_stereo_change(
        pm.StereoChange(
            reactant=pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1),
            product=pm.Tetrahedral(atoms=(0, 1, 2, 5, 4), parity=1),
            ts=pm.Tetrahedral(atoms=(0, 1, 2, 4, None)),
        )
    )
    return gra


@needs_smiles
@pytest.mark.parametrize("smiles", ["C", "CO", "C[C@@H](O)/C=C/C", "[CH2]", "[NH4+]"])
def test_round_trip(smiles: str) -> None:
    """from_json(to_json(g)) == g, and to_json is stable."""
    gra = pm.from_smiles(smiles)  # ty: ignore[missing-attribute]  Step 4
    string = pm.to_json(gra)
    assert pm.from_json(string) == gra
    assert pm.to_json(pm.from_json(string)) == string


def test_bytes_independent_of_insertion_order() -> None:
    """The same graph built in two orders gives the same bytes."""
    # trans-CHF=CHF: C0=C1, F2 and H3 on C0, F4 and H5 on C1
    gra1 = _build(pm.StereoMolGraph, "CCFHFH", [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])
    gra1.set_stereo(pm.PlanarBond(atoms=(2, 3, 0, 1, 5, 4), parity=0))
    gra1.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, None)))

    gra2 = pm.StereoMolGraph()
    for atom, symbol in reversed(list(enumerate("CCFHFH"))):
        gra2.add_node(atom, symbol=symbol)
    gra2.add_edges_from([(5, 1), (4, 1), (3, 0), (2, 0), (1, 0)])
    gra2.set_stereo(pm.Tetrahedral(atoms=(0, None, 3, 1, 2)))
    gra2.set_stereo(pm.PlanarBond(atoms=(4, 5, 1, 0, 3, 2), parity=0))

    assert pm.to_json(gra1) == pm.to_json(gra2)
    assert gra1 == gra2


def test_json_form() -> None:
    """Compact, the fields of the type in schema order, every LS field written."""
    gra = _stereo_sn2()
    assert json.loads(pm.to_json(gra)) == {
        "type": "StereoReactionGraph",
        "charge": -1,
        "spin": 0,
        "symbols": ["C", "F", "Cl", "Br", "H", "I"],
        "bonds": [[0, 1], [0, 2], [0, 4]],
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
                    "atoms": [0, 1, 2, 5, 4],
                    "parity": 1,
                },
                "ts": {
                    "kind": "Tetrahedral",
                    "atoms": [0, None, 1, 2, 4],
                    "parity": None,
                },
            }
        ],
    }
    assert " " not in pm.to_json(gra)
    assert list(json.loads(pm.to_json(_sn2(pm.ReactionGraph)))) == [
        "type",
        "charge",
        "spin",
        "symbols",
        "bonds",
        "breaking",
        "forming",
    ]
    mol = _build(pm.StereoMolGraph, "CC", [(0, 1)])
    mol.set_stereo(pm.PlanarBond(atoms=(None, None, 0, 1, None, None)))
    assert json.loads(pm.to_json(mol))["bond_stereo"] == [
        {
            "kind": "PlanarBond",
            "atoms": [None, None, 0, 1, None, None],
            "parity": None,
            "chain": [],
        }
    ]


def test_changes_sort_atom_sites_before_bond_sites() -> None:
    """Atom-site changes by centre, then bond-site changes by end pair."""
    # CH2=CH-CH3 → CH3-CH=CH2 (made-up, to have one atom and one bond change)
    gra = _build(
        pm.StereoReactionGraph,
        "CCCHHHHHH",
        [(0, 1), (0, 3), (0, 4), (1, 5), (2, 6), (2, 7), (2, 8), (1, 2)],
    )
    gra.set_stereo_change(
        pm.StereoChange(
            reactant=pm.PlanarBond(atoms=(3, 4, 0, 1, 2, 5), parity=0),
            product=pm.PlanarBond(atoms=(3, 4, 0, 1, 5, 2), parity=0),
        )
    )
    gra.set_stereo_change(
        pm.StereoChange(reactant=pm.Tetrahedral(atoms=(2, 1, 6, 7, 8), parity=1))
    )
    changes = json.loads(pm.to_json(gra))["stereo_changes"]
    assert [change["reactant"]["kind"] for change in changes] == [
        "Tetrahedral",
        "PlanarBond",
    ]
    assert pm.from_json(pm.to_json(gra)) == gra


@pytest.mark.parametrize(
    "gra",
    [
        pytest.param(_sn2(pm.ReactionGraph), id="ReactionGraph"),
        pytest.param(_stereo_sn2(), id="StereoReactionGraph"),
    ],
)
def test_reaction_types_round_trip(gra: AnyGraph) -> None:
    """Breaking and forming bonds and stereo changes survive the round trip."""
    copy = pm.from_json(pm.to_json(gra))
    assert type(copy) is type(gra)
    assert copy == gra


def test_unknown_type_raises() -> None:
    """The type must be one of the four."""
    string = '{"type":"Graph","charge":0,"spin":0,"symbols":[],"bonds":[]}'
    with pytest.raises(pm.GraphError):
        pm.from_json(string)


@pytest.mark.parametrize(
    "string",
    [
        pytest.param("not json", id="not-json"),
        pytest.param(
            '{"type":"MolGraph","charge":0,"spin":0,"symbols":["H","H"],'
            '"bonds":[[0,1]],"breaking":[]}',
            id="field-of-another-type",
        ),
        pytest.param(
            '{"type":"MolGraph","charge":0,"spin":0,"symbols":[],"bonds":[],"x":1}',
            id="unknown-field",
        ),
        pytest.param(
            '{"type":"MolGraph","charge":0,"spin":0,"symbols":["H","H"],'
            '"bonds":[[0,1],[1,0]]}',
            id="bond-twice",
        ),
        pytest.param(
            '{"type":"ReactionGraph","charge":0,"spin":0,"symbols":["H","H"],'
            '"bonds":[[0,1]],"breaking":[[0,1]]}',
            id="bond-in-two-lists",
        ),
        pytest.param(
            '{"type":"MolGraph","charge":0,"spin":0,"symbols":["H","H"],'
            '"bonds":[[0,2]]}',
            id="atom-out-of-range",
        ),
        pytest.param(
            '{"type":"MolGraph","charge":0,"spin":1,"symbols":["H","H"],"bonds":[]}',
            id="invalid-graph",
        ),
    ],
)
def test_bad_input_raises(string: str) -> None:
    """Malformed, foreign or invalid input raises GraphError."""
    with pytest.raises(pm.GraphError):
        pm.from_json(string)


def test_two_ls_on_one_site_raises() -> None:
    """The JSON lists each LS site once."""
    gra = _build(pm.StereoMolGraph, "CFXYH", [(0, 1), (0, 2), (0, 3), (0, 4)])
    gra.set_stereo(pm.Tetrahedral(atoms=(0, 1, 2, 3, 4), parity=1))
    data = json.loads(pm.to_json(gra))
    data["atom_stereo"] *= 2
    with pytest.raises(pm.GraphError, match="Two LS"):
        pm.from_json(json.dumps(data))


def test_to_json_needs_atoms_zero_to_n() -> None:
    """Symbols are written by index, so the atoms must be 0 to n - 1."""
    gra = pm.MolGraph()
    gra.add_node(1, symbol="H")
    with pytest.raises(pm.GraphError):
        pm.to_json(gra)


def test_two_changes_on_one_site_raises() -> None:
    """The JSON lists each changing site once."""
    data = json.loads(pm.to_json(_stereo_sn2()))
    data["stereo_changes"] *= 2
    with pytest.raises(pm.GraphError, match="Two changes"):
        pm.from_json(json.dumps(data))
