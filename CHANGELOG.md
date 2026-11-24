# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
