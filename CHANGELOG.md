# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `canonical_direction(gra) -> (graph, reversed)` for a `ReactionGraph` or
  `StereoReactionGraph`: `gra` or `reverse(gra)`, whichever has the smaller
  `canonical_json`, and whether it was reversed (a tie keeps `gra`). Atom indices are
  kept, so an atom map from `canonical_form` of the result applies to the input's atoms.
  A reaction graph and its reverse give the same result, so one TS imported in either
  direction gets one identity.

## [0.0.2] - 2026-11-24

### Added

- `canonical_form(gra) -> (canonical, atom_map)` for the four graph types, with
  `atom_map[k]` the input index of canonical atom k (a tuple), and `canonical_json(gra)`,
  the JSON of the canonical graph. The numbering is stereomolgraph's `canon_atom_num`;
  it doesn't see chain LS yet.
- `isomorphism(g1, g2)`: an atom map `m` with `relabel(g1, m) == g2`, or `None`
  (stereomolgraph's VF2++, comparing stereo, breaking and forming bonds, stereo changes,
  charge and spin), and `deduplicate(graphs)`: the first graph of each isomorphism class,
  in input order.
- `from_geometry(geo)`: the `StereoMolGraph` of an `autogeom.Geometry`, keeping its atom
  order, charge and spin. Bonds come from covalent radii and a Lewis structure from
  autogeom's solver. The LS sites are the low-T ones `from_smiles` gives, and every
  parity is read from the coordinates (undetermined only for a flat centre or a double
  bond twisted by about 90°).
- `embed(gra)`: an `autogeom.Geometry` of a `MolGraph` or `StereoMolGraph`, in its atom
  order, from RDKit's ETKDG on a perceived Lewis structure (autogeom's solver, so
  radicals work), with the determined LS as constraints. Every determined LS is checked
  on the result, so `from_geometry(embed(gra)) == gra` for a determined graph. Fixed
  seeds make it reproducible; fragments are set 3 Å apart.

### Changed

- protomol depends on autogeom 0.0.5 or later (from the `avcopan/automech` channel on
  prefix.dev), whose Lewis solver keeps onium octets, so `from_geometry` keeps the C=O⁺ and
  C=N⁺ `PlanarBond` of Criegee intermediates and nitrones.

- `ts_graphs` compares and deduplicates its channels with `isomorphism` and
  `deduplicate`.

### Fixed

- `AtropBond`'s permutation group: an end swap now keeps the helicity (the axis's C2),
  so a 1,3-disubstituted allene or a symmetric twisted TS no longer equals its mirror
  image. The normal form, and so the JSON, of an `AtropBond` may change. The vendored
  stereomolgraph now tracks the fork's `protomol` branch, which has the same fix.

## [0.0.1] - 2026-11-23

### Added

- The four graph types, `MolGraph`, `StereoMolGraph`, `ReactionGraph` and
  `StereoReactionGraph`: frozen pydantic models with keyword-only constructors that
  normalize and validate, and `AnyGraph`, their union discriminated by `type`.
- Local stereodescriptors: `Tetrahedral`, `SquarePlanar`, `TrigonalBipyramidal`,
  `Octahedral`, `PlanarBond` and `AtropBond` (with a `chain` for cumulated chains), and
  `StereoChange` for a reaction graph's changing sites; `invert`, `isomers` and `site`.
- `with_stereo`, `remove_stereo`, `stereo_at`, `neighbors` and `is_determined`.
- `to_json` and `from_json`: one compact, deterministic JSON string per graph.
- `from_smiles`, with every atom explicit in SMILES order, the low-T stereo scheme
  (trivial sites filled, unspecified ones left undetermined), and one LS per cumulated
  chain.
- `reaction_graph`, `reactants`, `products`, `reverse`, `relabel` and `without_stereo`.
- `ts_graphs`: the transition-state graphs of a reaction, fewest bonds first (ported from
  autoccv), each one elementary step that changes something: a degenerate reaction gives
  its bond-changing channels, and a stereoisomerization its one rotation or lone-pair
  inversion.
- `display` and Jupyter drawings of the four types.
- `protomol.examples`: `species`, `abstraction_ts` and `stereoisomerization_ts`.
