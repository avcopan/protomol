# protomol

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)

A draft of the graph API for [AutoStorage](https://github.com/avcopan/autostorage) and
AutoEngine: graph types for molecules and reactions with local stereodescriptors.

It has four immutable graph types (`MolGraph`, `StereoMolGraph`, `ReactionGraph`,
`StereoReactionGraph`), which are frozen pydantic models; their local stereodescriptors
(`Tetrahedral`, `PlanarBond`, `AtropBond`, …, and `StereoChange` for a reaction);
`from_smiles`, and `from_reaction_smiles` for atom-mapped reactions; `from_geometry` and `embed`, to and from autogeom geometries; a
deterministic JSON form; the transition-state graphs of a reaction (`ts_graphs`);
drawings in Jupyter; and three test graphs in `protomol.examples`.

## Installation

```bash
pip install protomol
```

or, in a pixi workspace, `pixi add --pypi protomol`. It is also on the `avcopan/automech`
channel of prefix.dev.

## Usage

```python
import protomol as pm

# Every H is explicit, and atom i of the graph is atom i of the SMILES
gra = pm.from_smiles("CC(O)CC")
gra.symbols[:4]  # ('C', 'C', 'O', 'C')

# In Jupyter: C1 is marked "?", since the SMILES doesn't give its configuration
pm.display(gra)

# One compact, deterministic JSON string per graph
text = pm.to_json(gra)
assert pm.from_json(text) == gra

# The transition-state graphs of a reaction, in the reactants' atom order
(ts,) = pm.ts_graphs(pm.from_smiles("CC.[OH]"), pm.from_smiles("C[CH2].O"))
ts.breaking, ts.forming  # ((0, 3),), ((2, 3),)
```

`protomol.examples` has three test graphs, each with its golden JSON in `tests/data/examples/`:

| function | graph |
|---|---|
| `species()` | (E)-pent-3-en-2-ol, one enantiomer (`C[C@@H](O)/C=C/C`) |
| `abstraction_ts()` | C2H6 + OH → C2H5 + H2O, H abstraction |
| `stereoisomerization_ts()` | (Z) → (E) but-2-en-1-yl radical, rotation about C2=C3 |

The pages in `user_guide/` show protomol as AutoEngine and AutoStorage use it, one page
per interface. Their `>>>` examples run as doctests; the planned examples show the API
still to come.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Credits

This package was created with [copier](https://copier.readthedocs.io/) and the [avcopan/copier-pixi-python](https://github.com/avcopan/copier-pixi-python) project template.
