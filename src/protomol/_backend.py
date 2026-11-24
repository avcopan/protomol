"""Conversion to and from stereomolgraph, the back end (P3, P7).

The only module that imports `protomol._vendor.stereomolgraph`; its types never leave
this module's private functions. Atom ids are the same ints in both, and the LS kinds
map 1:1 by name. A change's slots map to stereomolgraph's `Change`: `reactant` to
`BROKEN`, `product` to `FORMED` and `ts` to `FLEETING`.

stereomolgraph can't hold a chain LS (a cumulated chain), so `to_smg` drops one with a
warning.
"""

import warnings
from dataclasses import dataclass
from typing import Any

from ._stereo import (
    AtropBond,
    Octahedral,
    PlanarBond,
    SquarePlanar,
    StereoChange,
    Tetrahedral,
    TrigonalBipyramidal,
)
from ._types import (
    AnyGraph,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
)
from ._vendor.stereomolgraph import stereodescriptors as smg_ls
from ._vendor.stereomolgraph.algorithms.canon import canon_atom_num
from ._vendor.stereomolgraph.algorithms.circular import (
    color_refine_crg,
    color_refine_mg,
    color_refine_scrg,
    color_refine_smg,
    label_hash,
    numpy_int_multiset_hash,
)
from ._vendor.stereomolgraph.algorithms.isomorphism import vf2pp_all_isomorphisms
from ._vendor.stereomolgraph.graphs.crg import Change
from ._vendor.stereomolgraph.graphs.crg import CondensedReactionGraph as _SmgCRG
from ._vendor.stereomolgraph.graphs.mg import MolGraph as _SmgMG
from ._vendor.stereomolgraph.graphs.scrg import (
    StereoCondensedReactionGraph as _SmgSCRG,
)
from ._vendor.stereomolgraph.graphs.smg import StereoMolGraph as _SmgSMG
from ._vendor.stereomolgraph.periodic_table import SYMBOLS

type _LS = (
    Tetrahedral
    | SquarePlanar
    | TrigonalBipyramidal
    | Octahedral
    | PlanarBond
    | AtropBond
)

# protomol LS class → stereomolgraph LS class, matched by name
_LS_KINDS: tuple[type[_LS], ...] = (
    Tetrahedral,
    SquarePlanar,
    TrigonalBipyramidal,
    Octahedral,
    PlanarBond,
    AtropBond,
)
_TO_SMG_LS: dict[type[_LS], Any] = {
    cls: getattr(smg_ls, cls.__name__) for cls in _LS_KINDS
}
_FROM_SMG_LS: dict[Any, type[_LS]] = {
    getattr(smg_ls, cls.__name__): cls for cls in _LS_KINDS
}


def to_smg(gra: AnyGraph) -> Any:  # noqa: ANN401  (stereomolgraph is untyped here)
    """The stereomolgraph graph of `gra` (of the matching type).

    A chain LS is dropped with a warning: stereomolgraph can't hold one.
    """
    reaction = isinstance(gra, ReactionGraph | StereoReactionGraph)
    stereo = isinstance(gra, StereoMolGraph | StereoReactionGraph)
    cls: Any = {
        (False, False): _SmgMG,
        (False, True): _SmgSMG,
        (True, False): _SmgCRG,
        (True, True): _SmgSCRG,
    }[reaction, stereo]
    smg = cls()
    for i, symbol in enumerate(gra.symbols):
        smg.add_atom(i, symbol)
    _add_bonds(smg, gra)
    if isinstance(gra, StereoMolGraph | StereoReactionGraph):
        _add_stereo(smg, gra)
    return smg


def from_smg(
    smg: Any,  # noqa: ANN401  (stereomolgraph is untyped here)
    *,
    charge: int,
    spin: int,
) -> AnyGraph:
    """The protomol graph of a stereomolgraph graph, whose atoms are 0 to n - 1.

    Raises:
        ValueError: If the atoms are not 0 to n - 1, a bond is fleeting, or an LS is of
            a kind protomol doesn't have (pydantic's `ValidationError` for an invalid
            graph is a `ValueError` too).
    """
    atoms = sorted(smg.atoms)
    if atoms != list(range(len(atoms))):
        msg = f"stereomolgraph atoms must be 0 to n - 1, got {atoms}"
        raise ValueError(msg)
    data: dict[str, Any] = {
        "symbols": [SYMBOLS[smg.get_atom_type(i)] for i in atoms],
        "bonds": [tuple(bond) for bond in smg.bonds],
        "charge": charge,
        "spin": spin,
    }
    reaction = isinstance(smg, _SmgCRG)
    stereo = isinstance(smg, _SmgSMG)
    if reaction:
        if smg.get_fleeting_bonds():
            msg = "protomol has no fleeting bonds"
            raise ValueError(msg)
        data["breaking"] = [tuple(bond) for bond in smg.get_broken_bonds()]
        data["forming"] = [tuple(bond) for bond in smg.get_formed_bonds()]
    if stereo:
        data["atom_stereo"] = [_from_smg_ls(ls) for ls in smg.atom_stereo.values()]
        data["bond_stereo"] = [_from_smg_ls(ls) for ls in smg.bond_stereo.values()]
    if reaction and stereo:
        changes = [*smg.atom_stereo_changes.values(), *smg.bond_stereo_changes.values()]
        data["stereo_changes"] = [_from_smg_change(change) for change in changes]
        return StereoReactionGraph(**data)
    if reaction:
        return ReactionGraph(**data)
    if stereo:
        return StereoMolGraph(**data)
    return MolGraph(**data)


def canon_numbering(gra: AnyGraph) -> tuple[int, ...]:
    """A canonical atom order: canonical atom k is atom `order[k]` of `gra` (P22).

    The back-end's `canon_atom_num` on `to_smg(gra)`, as stereomolgraph gives it (P45).
    A chain LS isn't seen by the numbering (`to_smg` drops it with a warning), so two
    numberings of a graph with one may differ by a permutation that moves it.
    """
    if not gra.symbols:
        return ()
    numbers = canon_atom_num(to_smg(gra))
    return tuple(sorted(numbers, key=numbers.__getitem__))


@dataclass(frozen=True)
class Matchable:
    """A graph ready for VF2++: its back-end graph and colour refinement, built once."""

    smg: Any
    colors: Any  # the back-end's colour array (one int per atom)
    stereo: bool
    reaction: bool

    @property
    def fingerprint(self) -> int:
        """The multiset hash of the colours (the back-end's `__hash__`).

        Equal for isomorphic graphs; unequal graphs may collide. Not a canonical form.
        """
        if not len(self.colors):
            return 0
        return int(numpy_int_multiset_hash(self.colors))


def matchable(gra: AnyGraph) -> Matchable:
    """`gra` prepared for `vf2pp`, with the labels the back-end's `__eq__` uses."""
    reaction = isinstance(gra, ReactionGraph | StereoReactionGraph)
    stereo = isinstance(gra, StereoMolGraph | StereoReactionGraph)
    smg = to_smg(gra)
    if not gra.symbols:
        colors: Any = ()
    else:
        refine = {
            (False, False): color_refine_mg,
            (False, True): color_refine_smg,
            (True, False): color_refine_crg,
            (True, True): color_refine_scrg,
        }[reaction, stereo]
        labels = ("atom_type", "reaction") if reaction else ("atom_type",)
        colors = refine(smg, atom_labels=label_hash(smg, atom_labels=labels))
    return Matchable(smg=smg, colors=colors, stereo=stereo, reaction=reaction)


def vf2pp(first: Matchable, second: Matchable) -> tuple[int, ...] | None:
    """One isomorphism `m` with atom k of `second` = atom `m[k]` of `first`, or None.

    The back-end's `vf2pp_all_isomorphisms`, comparing the LS (stereo types) and the
    stereo changes (`StereoReactionGraph`), as the back-end's `__eq__` does. Breaking
    and forming bonds take part through the colours.
    """
    if len(first.colors) != len(second.colors):
        return None
    if not len(first.colors):
        return ()
    for iso in vf2pp_all_isomorphisms(
        first.smg,
        second.smg,
        atom_labels=(first.colors, second.colors),
        stereo=first.stereo,
        stereo_change=first.stereo and first.reaction,
        subgraph=False,
    ):
        inverse = {b: a for a, b in iso.items()}
        return tuple(inverse[k] for k in range(len(inverse)))
    return None


def _add_bonds(smg: Any, gra: AnyGraph) -> None:  # noqa: ANN401
    if isinstance(gra, MolGraph | StereoMolGraph):
        for bond in gra.bonds:
            smg.add_bond(*bond)
        return
    breaking, forming = set(gra.breaking), set(gra.forming)
    for bond in gra.bonds:
        if bond in breaking:
            smg.add_broken_bond(*bond)
        elif bond in forming:
            smg.add_formed_bond(*bond)
        else:
            smg.add_bond(*bond)


def _add_stereo(smg: Any, gra: StereoMolGraph | StereoReactionGraph) -> None:  # noqa: ANN401
    for ls in (*gra.atom_stereo, *gra.bond_stereo):
        smg_ls_ = _to_smg_ls(ls)
        if smg_ls_ is None:
            continue
        if isinstance(ls, PlanarBond | AtropBond):
            smg.set_bond_stereo(smg_ls_)
        else:
            smg.set_atom_stereo(smg_ls_)
    if isinstance(gra, StereoReactionGraph):
        for change in gra.stereo_changes:
            _set_change(smg, change)


def _to_smg_ls(ls: _LS) -> Any:  # noqa: ANN401
    """The stereomolgraph LS, or None (with a warning) for a chain LS."""
    if isinstance(ls, PlanarBond | AtropBond) and ls.chain:
        warnings.warn(
            f"stereomolgraph can't hold a chain LS; dropped {ls}",
            stacklevel=3,
        )
        return None
    return _TO_SMG_LS[type(ls)](ls.atoms, ls.parity)


def _from_smg_ls(ls: Any) -> _LS:  # noqa: ANN401
    cls = _FROM_SMG_LS.get(type(ls))
    if cls is None:
        msg = f"protomol has no LS kind for {ls!r}"
        raise ValueError(msg)
    return cls(atoms=ls.atoms, parity=ls.parity)


def _set_change(smg: Any, change: StereoChange) -> None:  # noqa: ANN401
    slots = {
        "broken": change.reactant,
        "formed": change.product,
        "fleeting": change.ts,
    }
    smg_slots = {name: _to_smg_ls(ls) for name, ls in slots.items() if ls is not None}
    smg_slots = {name: ls for name, ls in smg_slots.items() if ls is not None}
    if not smg_slots:
        return
    some = next(ls for ls in slots.values() if ls is not None)
    if isinstance(some, PlanarBond | AtropBond):
        smg.set_bond_stereo_change(**smg_slots)
    else:
        smg.set_atom_stereo_change(**smg_slots)


def _from_smg_change(change: Any) -> StereoChange:  # noqa: ANN401
    def slot(key: Change) -> _LS | None:
        ls = change.get(key)
        return None if ls is None else _from_smg_ls(ls)

    return StereoChange(
        reactant=slot(Change.BROKEN),
        product=slot(Change.FORMED),
        ts=slot(Change.FLEETING),
    )
