# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `enantiomer(g)`, `is_chiral(g)` and `canonical_enantiomer(g)` for the four graph
  types (P50, P60). `enantiomer` inverts every chiral LS (`Tetrahedral`, `AtropBond`,
  `TrigonalBipyramidal`, `Octahedral`, and a reaction graph's change slots), keeping
  planar bonds, undetermined LS and atom indices. `is_chiral` compares the canonical
  JSON of the two, a reaction graph's in its canonical direction (P75: N inversion,
  whose mirror image is its own reverse, is achiral). `canonical_enantiomer` returns whichever has the smaller canonical
  JSON (a tie keeps `g`); on a reaction graph each is compared in its canonical
  direction, so the choice doesn't depend on the direction.
- The canonical TS, documented in `user_guide/02-identity.qmd` and tested:
  `canonical_direction(canonical_enantiomer(ts))` is the same for a TS, its reverse,
  its mirror image and the mirror image reversed (P60). No new function.
- `from_reaction_smiles(smiles, *, charge=None, spin=None)` reads an atom-mapped
  reaction SMILES into a `StereoReactionGraph` (P62): atom k is the atom with the k-th
  smallest map number, each side is perceived as `from_smiles` does, and the `ts`
  slots are set as `reaction_graph` sets them. Every atom, H included, must be mapped, and both
  sides must have the same map numbers; the default spin is the lower of the two
  sides' radical counts.
- `stereoisomers(g)` for `MolGraph` and `StereoMolGraph` (P65): the distinct
  determined stereoisomers, both mirror images kept, with `g`'s atom indices. Every
  undetermined LS takes each configuration, determined ones are kept, and the results
  are deduplicated up to atom numbering; a `MolGraph` first gets its low-T sites. Gives
  StereoMolGraph SI-001's 10 stereoisomers. An allene's two collapse to one, as the
  back-end drops chain LS (P45). On a `ReactionGraph` (after `low_t`) or a
  `StereoReactionGraph` (P74), each change's side slots are expanded and its `ts` slot
  follows by the TS-slot rules, so the results are the channels: a rotation's two
  twisted TSs, an open SN2 centre's backside and frontside TSs and their mirror
  images. A bifurcation's `ts` slot stays undetermined (one channel per product).
- `transition_state(srg)` (P74, P5): where the TS-slot rules leave a change's `ts`
  slot undetermined with its sides determined (a bifurcation), the side slots become
  undetermined too, so a bifurcation's channels give one TS graph; any other graph is
  returned as it is.

### Changed

- TS sites (P67-P74, P80, P81): the `ts` slot of each change at a TS site (an LS with
  an atom in the reacting core: the atoms of the breaking and forming bonds, or of the
  changing site when no bond changes) is now set by rule, where it was always `None`.
  One side only: that side's LS, or undetermined when the other side has a triple
  bond there (a bifurcation); a centre that swaps a neighbour: the backside
  `TrigonalBipyramidal` (inversion) or a `Tetrahedral` with `None` where the leaving
  and attacking atoms meet (retention); a rotation: an `AtropBond`, both parities as
  two channels; a lone-pair inversion: `None`. `ts_graphs`, `reaction_graph`,
  `from_reaction_smiles` and `low_t` set it; constructors don't. A TS's graph, and so
  its canonical JSON, changes (`examples.abstraction_ts()` among them); species
  graphs don't. `ts_graphs` of a rotation now gives two channels.
- `geometry_mismatches` takes a reaction graph's geometry to be its TS's (P69, P82):
  it reads the determined `ts` slots too (a `TrigonalBipyramidal` as stereomolgraph
  reads one: the widest angle is axial), and its new `reacting_bonds` field lists the
  breaking and forming bonds that aren't partial (a full bond, up to 1.05 times the
  covalent-radius sum, or longer than twice it). An end point is checked on its side,
  `geometry_mismatches(products(ts), geo)`.
- `canonical_form`, `canonical_json`, `relabel` and `canonical_direction` accept a
  union argument (`AnyGraph`, or `ReactionGraph | StereoReactionGraph` for
  `canonical_direction`), which ty rejected before; a precise argument still gives the
  precise type back (P59). So do `reverse` and `remove_stereo` (P76). Typing only.

## [0.0.3] - 2026-11-25

### Added

- `low_t(g)` for `MolGraph | StereoMolGraph` (giving a `StereoMolGraph`) and
  `ReactionGraph | StereoReactionGraph` (giving a `StereoReactionGraph`): the low-T
  scheme's sites, perceived on a Lewis structure of the graph. A site the graph has
  keeps its LS and parity, a site the scheme drops loses its LS, and an added site is
  undetermined, or fixed if trivial. A reaction graph's sides are reassigned
  separately; each `StereoChange` slot follows its side's site, and the TS slots are
  kept. It is idempotent and a no-op on `from_smiles` and `from_geometry` output, but
  for a carbonyl-oxide radical whose perceived Lewis structure is a peroxy radical
  (open question 11).
- `canonical_direction(gra) -> (graph, reversed)` for a `ReactionGraph` or
  `StereoReactionGraph`: `gra` or `reverse(gra)`, whichever has the smaller
  `canonical_json`, and whether it was reversed (a tie keeps `gra`). Atom indices are
  kept, so an atom map from `canonical_form` of the result applies to the input's atoms.
  A reaction graph and its reverse give the same result, so one TS imported in either
  direction gets one identity.

### Changed

- `from_smiles` and `from_geometry` put a lone-pair `Tetrahedral` on exactly the
  pyramidal atoms of the low-T scheme: N, P or S with three neighbours and one lone pair
  (amines, phosphines, sulfoxides, sulfonium ions), but not an N next to a π system or
  a radical centre (amides, anilines, enamines, nitroxides, α-amino radicals). Phosphines
  and sulfoxides used to get one only from a chiral tag.
- Radicals with formal charges, and vinoxy radicals from their carbonyl form, get an LS
  on each resonating double bond (stereomolgraph's radical resonance, fixed on the fork's
  `protomol` branch).
- `from_smiles` keeps a chiral tag on a pyramidal N (`C[N@](F)Cl`), which RDKit drops.
- A chiral centre with a lone pair that starts its SMILES fragment reads the lone pair
  where an implicit H would go (Open Babel, CDK), not last (RDKit): `[N@](C)(F)Cl` is
  `C[N@@](F)Cl`. This flips atom-first N, P and S centres.
- An acyclic double bond or cumulated chain with a radical end (an alkenyl or allenyl
  radical) gets no LS: the low-T scheme lumps its cis and trans forms.
- A site is filled as trivial only if it is trivial for every configuration of the other
  undetermined sites: the centres of 1,4-dimethylcyclohexane (cis/trans) and the
  pseudoasymmetric C of 2,3,4-trihydroxyglutaric acid stay undetermined.
- Every site InChI keeps is a low-T site: arsines, selenoxides and selenonium ions get a
  lone-pair `Tetrahedral`, and so does an N in a three-membered ring even when it is
  conjugated (N-acyl, N-aryl aziridines). `embed` makes such a centre pyramidal when
  ETKDG leaves it flat.
- A σ-radical C in a three-membered ring gets a `Tetrahedral` with its unpaired
  electron in the `None` slot if the ring has a heteroatom (oxiranyl, aziridinyl) or the
  C bears an N, O, F or Cl (α-fluoro cyclopropyl); a plain cyclopropyl radical does not.
  `from_smiles` keeps a chiral tag on such a centre (`C[C@H]1O[C@]1C`).
- An alkenyl or allenyl radical whose radical C bears an N, O, F or Cl keeps its bond or
  chain LS (`F[C]=CC`), with the unpaired electron in a `None` slot.
- A single bond with a closed-shell partial double bond gets a `PlanarBond`: an N with
  three neighbours (not in a three-membered ring) on a C=O, C=S or N=O (amides,
  thioamides, carbamates, ureas, imides, hydroxamic acids, nitrosamines), and the C-C of
  a vinyl carbonyl oxide (acrolein oxide's s-cis/s-trans). A ring of seven or fewer
  atoms holds it cis (lactams, caprolactam included). Esters, amidines, enamines and
  enones get none.
- A ring of seven atoms holds a double bond cis, as InChI does: a cycloheptene C=C is
  trivial, not undetermined (stereomolgraph's ring cutoff, fixed on the fork's
  `protomol` branch).

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
