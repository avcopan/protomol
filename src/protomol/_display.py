"""Drawing graphs with stereomolgraph's `View2D` (RDKit `MolDraw2DSVG`) (P8).

`View2D` perceives bond orders, draws the LS (wedges, and a crossed bond for an
undetermined bond LS), and colours a reaction graph's changes: breaking bonds red,
forming bonds blue, a bond whose stereo changes magenta. Two additions here: the graph's
`charge` goes to bond-order perception (`_charge`), and a private subclass puts a "?"
note on the centre of an undetermined atom LS (RDKit draws a plain centre otherwise).
"""

from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from typing import Any

from ._backend import to_smg
from ._types import AnyGraph
from ._vendor.stereomolgraph import ipython as _ipython
from ._vendor.stereomolgraph.ipython import View2D

# The converters `View2D._to_mol` looks up in its module, by name
_TO_RDMOL_NAMES = (
    "mol_graph_to_rdmol",
    "stereo_mol_graph_to_rdmol",
    "condensed_reaction_graph_to_rdmol",
    "stereo_condensed_reaction_graph_to_rdmol",
)


def display(gra: AnyGraph, *, atom_indices: bool = False) -> None:
    """Show the graph in Jupyter.

    The drawing shows the perceived bond orders and charges, the LS (undetermined ones
    marked), and for a reaction graph its breaking and forming bonds on the reactant
    structure: breaking bonds red, forming bonds blue, and a bond whose stereo changes
    magenta. An undetermined atom LS has a "?" on its centre, and an undetermined bond
    LS is drawn as a crossed double bond. Each graph type also draws itself in Jupyter
    (`_repr_svg_`), without the atom indices.

    Known limits: bond-order perception ignores `spin` (it is a port of xyz2mol), the
    perceived radicals aren't drawn, and a chain LS (a cumulated chain) isn't drawn.

    Args:
        gra: The graph.
        atom_indices: Label each atom with its index.

    Raises:
        ImportError: If IPython isn't installed.
    """
    # IPython is optional, and not in the type-checking environment
    import IPython.display  # noqa: PLC0415  # ty: ignore[import-not-found]

    IPython.display.display(IPython.display.SVG(_svg(gra, atom_indices=atom_indices)))


def _svg(gra: AnyGraph, *, atom_indices: bool = False) -> str:
    """The SVG drawing of `gra` (see `display`)."""
    view = _View2D(show_atom_numbers=atom_indices, generate_bond_orders=True)
    with _charge(gra.charge):
        return view.svg(to_smg(gra))


class _View2D(View2D):
    """`View2D`, with a "?" note on the centre of each undetermined atom LS."""

    def _to_mol(self, graph: Any) -> Any:  # noqa: ANN401  (stereomolgraph is untyped here)
        mol, highlights = super()._to_mol(graph)
        for centre in _undetermined_centres(graph):
            atom = mol.GetAtomWithIdx(centre)  # RDKit indices are the atom indices
            note = atom.GetProp("atomNote") if atom.HasProp("atomNote") else ""
            atom.SetProp("atomNote", f"{note}?")
        return mol, highlights


def _undetermined_centres(smg: Any) -> list[int]:  # noqa: ANN401
    """The centres of the undetermined atom LS (before or after a change)."""
    if not hasattr(smg, "atom_stereo"):
        return []
    centres = [c for c, ls in smg.atom_stereo.items() if ls.parity is None]
    for c, change in getattr(smg, "atom_stereo_changes", {}).items():
        if any(ls is not None and ls.parity is None for ls in change.values()):
            centres.append(c)
    return centres


@contextmanager
def _charge(charge: int) -> Generator[None]:
    """Pass `charge` to `View2D`'s bond-order perception, which has no option for it.

    xyz2mol uses the total charge only with charged fragments allowed, and then it
    also puts formal charges on a neutral radical, so these are on only for a charged
    graph.
    """
    if charge == 0:
        yield
        return
    originals = {name: getattr(_ipython, name) for name in _TO_RDMOL_NAMES}
    try:
        for name, func in originals.items():
            setattr(
                _ipython,
                name,
                partial(func, allow_charged_fragments=True, charge=charge),
            )
        yield
    finally:
        for name, func in originals.items():
            setattr(_ipython, name, func)
