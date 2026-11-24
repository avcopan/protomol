"""TS channels by the constructive count vector (CCV) algorithm (P18).

Ported from autoccv's `ccv.py` (fewest-bonds-first CCV): only the graph structure
changes. The search runs on unfrozen networkx copies of `_nx(gra)`, with `symbol` on
the nodes and, for reaction graphs, `change` on the edges.
"""

import functools
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import more_itertools as mit
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from rdkit import Chem

from ._backend import to_smg
from ._reaction import products as _products
from ._reaction import reactants as _reactants
from ._reaction import reaction_graph, relabel, reverse
from ._stereo import StereoChange, isomers
from ._types import (
    _LS,
    AnyGraph,
    GraphError,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
    _nx,
    _site,
    with_stereo,
)

type _Pair = tuple[int, int]
type _BondSymbol = tuple[str, str]

_PERIODIC_TABLE = Chem.GetPeriodicTable()


@overload
def ts_graphs(
    reactants: StereoMolGraph, products: StereoMolGraph, *, extra_breaks: int = 2
) -> list[StereoReactionGraph]: ...
@overload
def ts_graphs(
    reactants: MolGraph, products: MolGraph, *, extra_breaks: int = 2
) -> list[ReactionGraph]: ...
def ts_graphs(
    reactants: MolGraph | StereoMolGraph,
    products: MolGraph | StereoMolGraph,
    *,
    extra_breaks: int = 2,
) -> list[ReactionGraph] | list[StereoReactionGraph]:
    """The topologically distinct TS channels from reactants to products (CCV).

    Each graph is in the reactants' atom order. With stereo inputs, the channels are
    the stereo-distinct ones whose two sides are the given reactants and products
    (stereo-aware isomorphism): each topological channel's LS are enumerated at the
    sites its reaction touches, so diastereotopic atoms give separate channels (the two
    H of C3 in (R)-butan-2-ol, say). Every other LS stays unchanged; only if no
    enumeration matches the inputs may one of them change, then two, and so on (least
    motion), so a (Z) → (E) rotation keeps its one change and an unreacting CH3 is
    never inverted. The `ts` slots stay empty: fleeting stereo comes in Phase 5.

    `extra_breaks` is the most bonds the search breaks and re-forms beyond those the
    bond counts require.

    Raises:
        TypeError: If the two graphs are not both `MolGraph`s or both
            `StereoMolGraph`s.
        GraphError: If their charge or spin differ.
    """
    if not (
        type(reactants) is type(products)
        and isinstance(reactants, MolGraph | StereoMolGraph)
    ):
        msg = (
            "Expected two MolGraphs or two StereoMolGraphs, got "
            f"{type(reactants).__name__} and {type(products).__name__}"
        )
        raise TypeError(msg)
    for name in ("charge", "spin"):
        if getattr(reactants, name) != getattr(products, name):
            msg = (
                f"Reactants and products differ in {name}: "
                f"{getattr(reactants, name)} and {getattr(products, name)}"
            )
            raise GraphError(msg)
    results = _CCV(reactants, products).results(extra_breaks=extra_breaks)
    if isinstance(reactants, MolGraph):
        return [gra for gra, _ in results]
    assert isinstance(products, StereoMolGraph)  # noqa: S101 (for the type checker)
    n = len(reactants.symbols)
    return [
        srg
        for _, mapping in results
        for srg in _stereo_channels(
            reactants,
            products,
            reaction_graph(
                reactants, relabel(products, [mapping[k] for k in range(n)])
            ),
        )
    ]


def _stereo_channels(
    reactants: StereoMolGraph, products: StereoMolGraph, srg: StereoReactionGraph
) -> list[StereoReactionGraph]:
    """The stereo-distinct versions of one channel whose sides match the inputs.

    `srg` is the channel with one atom mapping. The changes at the sites its reaction
    touches (its centre, or a bond's end, is in a breaking or forming bond) get every
    configuration on each side; a side whose LS is undetermined stays undetermined.
    The other sites keep their reactant LS on both sides, unless no enumeration
    matches, in which case the fewest of them get another product LS (`_inverted`);
    `srg` itself bounds that number.
    Candidates are deduplicated, as in stereomolgraph's `generate_stereoisomers`, by
    frozen stereomolgraph graphs, whose hash and `==` are stereo-aware isomorphism.
    """
    reacting = {atom for bond in (*srg.breaking, *srg.forming) for atom in bond}
    touched: list[list[StereoChange]] = []
    untouched: list[_LS] = []
    for item in (*srg.atom_stereo, *srg.bond_stereo, *srg.stereo_changes):
        if isinstance(item, StereoChange):
            if _touches(item, reacting):
                touched.append(_change_configurations(item))
            elif item.reactant is not None:
                untouched.append(item.reactant)
        else:
            untouched.append(item)

    # `srg` itself is a candidate with this many inversions, so the search ends there
    most = sum(
        not _touches(item, reacting)
        for item in srg.stereo_changes
        if item.reactant is not None
    )
    want = (_frozen_smg(reactants), _frozen_smg(products))
    for count in range(most + 1):
        found: dict[object, StereoReactionGraph] = {}
        for inverted in _inverted(untouched, count):
            sites = {_site(change) for change in inverted}
            kept = [ls for ls in untouched if _site(ls) not in sites]
            for changes in itertools.product(*touched):
                cand = with_stereo(srg, *kept, *inverted, *changes)
                sides = (_frozen_smg(_reactants(cand)), _frozen_smg(_products(cand)))
                if sides == want:
                    found.setdefault(_frozen_smg(cand), cand)
        if found:
            return list(found.values())
    return []


def _touches(change: StereoChange, reacting: set[int]) -> bool:
    """Whether a change is structural, not a pure inversion.

    It is if its site (its centre, or a bond's ends) is a reacting atom, or if its LS
    has no counterpart on the other side, or other atoms there (a double bond that
    moves, say).
    """
    r, p = change.reactant, change.product
    if r is None or p is None or set(r.atoms) != set(p.atoms):
        return True
    site = _site(change)
    return any(
        atom in reacting for atom in ((site,) if isinstance(site, int) else site)
    )


def _change_configurations(change: StereoChange) -> list[StereoChange]:
    """Every configuration of a change's reactant and product LS (`ts` kept).

    A side whose LS is undetermined keeps it undetermined.
    """
    sides = [
        [ls] if ls is None or ls.parity is None else list(isomers(ls))
        for ls in (change.reactant, change.product)
    ]
    return [
        StereoChange(reactant=r, product=p, ts=change.ts)
        for r, p in itertools.product(*sides)
    ]


def _inverted(untouched: list[_LS], count: int) -> Iterator[list[StereoChange]]:
    """Each way to give `count` untouched LS another product configuration.

    Undetermined LS are never changed.
    """
    changeable = [ls for ls in untouched if ls.parity is not None]
    for chosen in itertools.combinations(changeable, count):
        others = [[iso for iso in isomers(ls) if iso != ls] for ls in chosen]
        for alternatives in itertools.product(*others):
            yield [
                StereoChange(reactant=ls, product=alt)
                for ls, alt in zip(chosen, alternatives, strict=True)
            ]


def _frozen_smg(gra: AnyGraph) -> object:
    """The frozen stereomolgraph graph (hash and `==`: stereo-aware isomorphism)."""
    smg = to_smg(gra)
    smg.freeze()
    return smg


@dataclass(frozen=True)
class _CCV:
    """The CCV search from reactants to products (autoccv's `CCV`)."""

    reactants: MolGraph | StereoMolGraph
    products: MolGraph | StereoMolGraph

    @functools.cached_property
    def reactant_bond_symbols(self) -> dict[_Pair, _BondSymbol]:
        return _bond_symbols(self.reactants)

    @functools.cached_property
    def product_bond_symbols(self) -> dict[_Pair, _BondSymbol]:
        return _bond_symbols(self.products)

    @functools.cached_property
    def reactant_bond_count_vector(self) -> Counter[_BondSymbol]:
        return Counter(self.reactant_bond_symbols.values())

    @functools.cached_property
    def product_bond_count_vector(self) -> Counter[_BondSymbol]:
        return Counter(self.product_bond_symbols.values())

    def results(
        self,
        *,
        extra_breaks: int = 2,
        unique_isomorphs: bool = True,
        unique_bond_changes: bool = True,
        maximum_bond_score: bool = True,
    ) -> list[tuple[ReactionGraph, dict[int, int]]]:
        """The TS graphs and their mappings (reactant atom → product atom), filtered.

        The filters keep the isomorphically distinct graphs, the graphs with distinct
        bond changes, and the graphs with the highest bonding score (the most open
        valences used: radical sites and multiple bonds).
        """
        max_score = None
        if maximum_bond_score:
            max_score = max(
                (
                    _bonding_score(gra)
                    for gra, _ in self.all_results(
                        extra_breaks=extra_breaks, unique_bond_changes=True
                    )
                ),
                default=None,
            )

        kept: list[tuple[ReactionGraph, dict[int, int]]] = []
        for gra, mapping in self.all_results(
            extra_breaks=extra_breaks, unique_bond_changes=unique_bond_changes
        ):
            if maximum_bond_score and _bonding_score(gra) != max_score:
                continue
            if unique_isomorphs and any(_is_isomorphic(gra, g) for g, _ in kept):
                continue
            kept.append((gra, mapping))
        return kept

    def all_results(
        self, *, extra_breaks: int = 2, unique_bond_changes: bool = False
    ) -> Iterator[tuple[ReactionGraph, dict[int, int]]]:
        """Every result with the fewest extra broken bonds that gives any."""
        seen: list[tuple[frozenset[_Pair], frozenset[_Pair]]] | None = (
            [] if unique_bond_changes else None
        )
        for num_extra in range(extra_breaks + 1):
            for extra_cv in self._extra_breaking_bond_count_vectors(num_extra):
                results = itertools.chain.from_iterable(
                    self._transition_graphs(b1, b2, seen)
                    for b1, b2 in self._breaking_bond_patterns(extra_cv)
                )
                first = next(results, None)
                if first is None:
                    continue
                yield first
                yield from results
                return

    def _extra_breaking_bond_count_vectors(
        self, num: int
    ) -> Iterator[Counter[_BondSymbol]]:
        rcv = self.reactant_bond_count_vector
        pcv = self.product_bond_count_vector
        cv = rcv - (rcv - pcv)
        symbs = sorted(cv.keys(), key=lambda x: (-x.count("H"), x))
        pool = list(
            itertools.chain.from_iterable(itertools.repeat(s, cv[s]) for s in symbs)
        )
        return map(Counter, mit.unique_everseen(itertools.combinations(pool, num)))

    def _breaking_bond_patterns(
        self, extra_cv: Counter[_BondSymbol]
    ) -> Iterator[tuple[tuple[_Pair, ...], tuple[_Pair, ...]]]:
        rcv0 = self.reactant_bond_count_vector
        pcv0 = self.product_bond_count_vector
        rcv = (rcv0 - pcv0) + extra_cv
        pcv = (pcv0 - rcv0) + extra_cv
        return itertools.product(
            _count_vector_bond_combinations(rcv, self.reactant_bond_symbols),
            _count_vector_bond_combinations(pcv, self.product_bond_symbols),
        )

    def _transition_graphs(
        self,
        break_bonds1: Sequence[_Pair],
        break_bonds2: Sequence[_Pair],
        seen: list[tuple[frozenset[_Pair], frozenset[_Pair]]] | None,
    ) -> Iterator[tuple[ReactionGraph, dict[int, int]]]:
        """The TS graphs whose broken reactants match the broken products."""
        gra1 = nx.Graph(_nx(self.reactants))
        gra1.remove_edges_from(break_bonds1)
        gra2 = nx.Graph(_nx(self.products))
        gra2.remove_edges_from(break_bonds2)
        for mapping in _matcher(gra1, gra2).isomorphisms_iter():
            inverse = {v: k for k, v in mapping.items()}
            breaking = frozenset(break_bonds1)
            forming = frozenset(
                (min(inverse[i], inverse[j]), max(inverse[i], inverse[j]))
                for i, j in break_bonds2
            )
            if seen is not None:
                if (breaking, forming) in seen:
                    continue
                seen.append((breaking, forming))
            gra = ReactionGraph(
                symbols=self.reactants.symbols,
                bonds={*self.reactants.bonds, *forming},
                charge=self.reactants.charge,
                spin=self.reactants.spin,
                breaking=breaking,
                forming=forming,
            )
            yield gra, mapping


def _bond_symbols(gra: MolGraph | StereoMolGraph) -> dict[_Pair, _BondSymbol]:
    """The sorted pair of element symbols of each bond."""
    s = gra.symbols
    return {(i, j): (min(s[i], s[j]), max(s[i], s[j])) for i, j in gra.bonds}


def _count_vector_bond_combinations(
    cv: Counter[_BondSymbol], symbs: dict[_Pair, _BondSymbol]
) -> Iterator[tuple[_Pair, ...]]:
    """The bond combinations with the counts of `cv`."""
    keys: defaultdict[_BondSymbol, list[_Pair]] = defaultdict(list)
    for key, symb in symbs.items():
        keys[symb].append(key)
    per_symbol = [itertools.combinations(keys[s], count) for s, count in cv.items()]
    for combination in itertools.product(*per_symbol):
        yield tuple(itertools.chain.from_iterable(combination))


def _open_valences(gra: MolGraph, atoms: Sequence[int]) -> list[int]:
    """The default valence (RDKit's) minus the degree, for each atom."""
    nxg = _nx(gra)
    return [
        _PERIODIC_TABLE.GetDefaultValence(gra.symbols[i]) - nxg.degree[i] for i in atoms
    ]


def _forward_bonding_score(gra: ReactionGraph) -> int:
    """The open valences on the reactant side of the atoms of the forming bonds."""
    atoms = sorted(set(itertools.chain.from_iterable(gra.forming)))
    return sum(_open_valences(_reactants(gra), atoms))


def _bonding_score(gra: ReactionGraph) -> int:
    return _forward_bonding_score(gra) + _forward_bonding_score(reverse(gra))


def _matcher(gra1: nx.Graph, gra2: nx.Graph) -> GraphMatcher:
    """A matcher on the `symbol` of nodes and the `change` of edges."""
    return GraphMatcher(
        gra1,
        gra2,
        node_match=lambda a, b: a["symbol"] == b["symbol"],
        edge_match=lambda a, b: a.get("change") == b.get("change"),
    )


def _is_isomorphic(gra1: ReactionGraph, gra2: ReactionGraph) -> bool:
    return _matcher(_nx(gra1), _nx(gra2)).is_isomorphic()
