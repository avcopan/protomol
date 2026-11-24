# protomol

protomol holds the graph types (molecules and reactions, with local stereodescriptors) that
AutoStorage and AutoEngine use. It stands in for the Python API of stereomol, a Rust
library still being written, so its public API is a draft of stereomol's: treat any new or
changed public name, type, field or signature as an API decision for the maintainer, not
something to settle in code. Private helpers (`_name`) are fine.

- **stereomolgraph is a private back-end.** A copy of its `canon` branch is vendored in
  `src/protomol/_vendor/stereomolgraph`, generated from the git-subrepo in
  `extern/stereomolgraph` by `pixi run vendor` (checked in pre-commit; never edit the copy).
  Its types never leave `_backend.py` (`to_smg`, `from_smg`): no public function takes or
  returns one. The other private modules may import its algorithms (`_smiles` its RDKit
  reader, `_display` its `View2D`), and the import-linter layers in `pyproject.toml` hold
  the order.
- **Index alignment.** Atom i of a graph is atom i of the geometry or SMILES it came from:
  `from_smiles` keeps the SMILES atoms in order, `[H]` atoms where they are written, then
  the implicit H in the order of the atoms they're bonded to. Every function keeps atom
  indices unless it says otherwise (`relabel`, and later the canonical form, return a
  relabelled graph and the atom map).
- **The graphs are frozen pydantic models.** Build every new graph through its
  constructor (which normalizes and validates); never use `model_copy(update=...)` or
  `model_construct`, which skip validation.
- `user_guide/*.qmd` are the use-case pages: `>>>` blocks are live doctests (pytest
  `--doctest-glob=*.qmd`), and plain code blocks under a "Planned" callout show API still to
  come. A change that implements a planned call turns its block into a live `>>>` block, and a
  change to a public name or signature updates the pages in the same commit.
