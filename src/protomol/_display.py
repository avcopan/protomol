"""Drawing graphs: an RDKit molecule built from the graph, drawn by `MolDraw2D` (P8).

The molecule is the graph's Lewis structure from the private `_lewis` (P43, P49), in the
graph's atom order, so it shows bond orders, formal charges and radicals; a reaction
graph's is the reactant side's, with the forming bonds added as dashed bonds. RDKit
lays out 2D coordinates; protomol's own LS then set the stereo marks:

- A determined atom LS gets a wedge or hash on one neighbour bond, chosen by lifting
  that neighbour out of the page and reading the parity back with `from_geometry`'s
  readers, so the drawing follows protomol's convention, not RDKit's chiral tags.
- A determined `AtropBond` (a rigid single bond, or an allene axis with its `chain`)
  gets a wedge or hash on one substituent of `end2`, the same way.
- A site with twin terminal groups (a CH3 or CH2 carbon, a =CH2 end) is marked only
  with the atom indices drawn: without them, its configurations look the same.
- A determined `PlanarBond` is laid out cis or trans: if RDKit's layout has it the
  wrong way, the `end2` side is reflected across the bond axis.
- An undetermined bond LS is a crossed double bond, or a "?" bond note on a chain's
  middle bond or a single bond; an undetermined atom LS is a "?" atom note.

A reaction graph's breaking bonds are red, forming bonds blue, and a stereo change's
site magenta (its bond or chain, or its centre atom).
"""

import itertools
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from ._geometry import (
    _atrop_bond_from_coords,
    _planar_bond_from_coords,
    _tetrahedral_from_coords,
)
from ._lewis import _lewis
from ._smiles import _has_twin_terminal_groups
from ._stereo import AtropBond, PlanarBond, Stereo, Tetrahedral
from ._types import (
    AnyGraph,
    GraphError,
    MolGraph,
    ReactionGraph,
    StereoMolGraph,
    StereoReactionGraph,
)

type _Pair = tuple[int, int]
type _Color = tuple[float, float, float]

_SIZE = 300  # px, width and height
_LIFT = 1.0  # out-of-page offset (2D bond lengths are 1.5) when choosing a wedge
_RED: _Color = (1.0, 0.0, 0.0)
_BLUE: _Color = (0.0, 0.0, 1.0)
_MAGENTA: _Color = (1.0, 0.0, 1.0)


def display(gra: AnyGraph, *, atom_indices: bool = False) -> None:
    """Show the graph in Jupyter.

    The drawing shows a Lewis structure (bond orders, formal charges and radicals,
    perceived with the graph's charge and spin), the LS as wedges (undetermined ones
    marked; a CH3 or CH2 carbon's only with `atom_indices`), and for a reaction graph
    its breaking and forming bonds on the reactant structure: breaking bonds red,
    forming bonds blue (dashed), and the site of a stereo change magenta. An
    undetermined atom LS has a "?" on its centre, and an undetermined bond LS is drawn
    as a crossed double bond (a cumulated chain's has a "?" on its middle bond). Each
    graph type also draws itself in Jupyter (`_repr_svg_`), without the atom indices.

    Known limit: of a species' resonance structures (allyl, acetate), one is drawn.

    Args:
        gra: The graph.
        atom_indices: Label each atom with its index.

    Raises:
        ImportError: If IPython isn't installed.
    """
    # IPython is optional, and not in the type-checking environment
    import IPython.display  # noqa: PLC0415

    IPython.display.display(IPython.display.SVG(_svg(gra, atom_indices=atom_indices)))


def _svg(gra: AnyGraph, *, atom_indices: bool = False) -> str:
    """The SVG drawing of `gra` (see `display`)."""
    return _draw(gra, rdMolDraw2D.MolDraw2DSVG(_SIZE, _SIZE), atom_indices=atom_indices)


def _draw(gra: AnyGraph, drawer: Any, *, atom_indices: bool) -> Any:  # noqa: ANN401  (RDKit is untyped)
    """Draw `gra` with an RDKit `MolDraw2D` and return its drawing text."""
    mol, bond_colors, atom_colors = _drawing_mol(gra, atom_indices=atom_indices)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.continuousHighlight = False
    options.highlightBondWidthMultiplier = 12
    options.fillHighlights = False
    options.prepareMolsBeforeDrawing = False  # keep the wedges set here
    bonds = {
        mol.GetBondBetweenAtoms(*pair).GetIdx(): color
        for pair, color in bond_colors.items()
    }
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bonds),
        highlightBondColors=bonds,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _drawing_mol(
    gra: AnyGraph, *, atom_indices: bool
) -> tuple[Chem.Mol, dict[_Pair, _Color], dict[int, _Color]]:
    """The molecule to draw, its bond colours (by atom pair) and atom colours."""
    reaction = isinstance(gra, ReactionGraph | StereoReactionGraph)
    breaking = set(gra.breaking) if reaction else set()
    forming = set(gra.forming) if reaction else set()
    lss, changes = _reactant_stereo(gra)

    mol = Chem.RWMol(_reactant_lewis(gra, forming))
    for i, j in sorted(forming):
        mol.AddBond(i, j, Chem.BondType.HYDROGEN)  # drawn dashed
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    mol.UpdatePropertyCache(strict=False)

    _set_planar_stereo(mol, lss)
    rdDepictor.Compute2DCoords(mol, clearConfs=True, sampleSeed=42, nSample=100)
    rdDepictor.StraightenDepiction(mol)
    _fix_planar_bonds(mol, lss)
    marked = [
        ls for ls in lss if atom_indices or not _has_twin_terminal_groups(gra, ls)
    ]
    _set_wedges(mol, _wedges(mol, marked, avoid=breaking | forming))
    notes = _undetermined_marks(mol, lss)
    for i in range(mol.GetNumAtoms()):
        text = (f"{i}" if atom_indices else "") + ("?" if i in notes else "")
        if text:
            mol.GetAtomWithIdx(i).SetProp("atomNote", text)

    bond_colors: dict[_Pair, _Color] = dict.fromkeys(breaking, _RED)
    bond_colors |= dict.fromkeys(forming, _BLUE)
    atom_colors: dict[int, _Color] = {}
    for ls in changes:
        if isinstance(ls, PlanarBond | AtropBond):
            for pair in _chain_bonds(ls):
                bond_colors.setdefault(pair, _MAGENTA)
        else:
            atom_colors[ls.atoms[0]] = _MAGENTA
    return mol, bond_colors, atom_colors


def _reactant_lewis(gra: AnyGraph, forming: set[_Pair]) -> Chem.Mol:
    """A Lewis structure of `gra`'s reactant side (all of `gra` for a species).

    If none fits (a reaction graph with the products' spin), single bonds, no charges.
    """
    bonds = [bond for bond in gra.bonds if bond not in forming]
    side = MolGraph(symbols=gra.symbols, bonds=bonds, charge=gra.charge, spin=gra.spin)
    try:
        return _lewis(side)
    except GraphError:
        mol = Chem.RWMol()
        for symbol in gra.symbols:
            atom = Chem.Atom(symbol)
            atom.SetNoImplicit(True)  # noqa: FBT003  (RDKit's API)
            mol.AddAtom(atom)
        for i, j in bonds:
            mol.AddBond(i, j, Chem.BondType.SINGLE)
        return mol.GetMol()


def _reactant_stereo(gra: AnyGraph) -> tuple[list[Stereo], list[Stereo]]:
    """The reactant side's LS, and one LS of each change (to colour its site).

    A change with no reactant LS (stereo made in the reaction) is coloured but gets no
    stereo mark.
    """
    if isinstance(gra, MolGraph | ReactionGraph):
        return [], []
    lss: list[Stereo] = [*gra.atom_stereo, *gra.bond_stereo]
    if isinstance(gra, StereoMolGraph):
        return lss, []
    changes: list[Stereo] = []
    for change in gra.stereo_changes:
        if change.reactant is not None:
            lss.append(change.reactant)
        changes.append(
            next(ls for ls in (change.reactant, change.ts, change.product) if ls)
        )
    return lss, changes


def _chain_bonds(ls: PlanarBond | AtropBond) -> list[_Pair]:
    """The bonds from end1 through the chain to end2, as sorted pairs."""
    _, _, end1, end2, _, _ = ls.atoms
    path = [end1, *ls.chain, end2]
    return [(min(i, j), max(i, j)) for i, j in itertools.pairwise(path)]


# Planar bonds: RDKit lays out a double bond's cis/trans bond stereo; a chain or a
# radical's bond it lays out freely, so each is checked and fixed after the layout


def _set_planar_stereo(mol: Chem.RWMol, lss: Iterable[Stereo]) -> None:
    """Cis/trans bond stereo on each determined `PlanarBond` of a double bond."""
    for ls in lss:
        if not isinstance(ls, PlanarBond) or ls.parity is None or ls.chain:
            continue
        a, b, end1, end2, c, d = ls.atoms
        bond = mol.GetBondBetweenAtoms(end1, end2)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        x, x_flipped = (a, False) if a is not None else (b, True)
        y, y_flipped = (c, False) if c is not None else (d, True)
        if x is None or y is None:
            continue
        if bond.GetBeginAtomIdx() != end1:
            x, y = y, x
        bond.SetStereoAtoms(x, y)
        bond.SetStereo(
            Chem.BondStereo.STEREOCIS
            if x_flipped == y_flipped
            else Chem.BondStereo.STEREOTRANS
        )


def _fix_planar_bonds(mol: Chem.RWMol, lss: Iterable[Stereo]) -> None:
    """Reflect the `end2` side of each determined `PlanarBond` laid out the wrong way.

    A bond in a ring is left as RDKit laid it out (the ring holds it).
    """
    conf = mol.GetConformer()
    for ls in lss:
        if not isinstance(ls, PlanarBond) or ls.parity is None:
            continue
        xyz = conf.GetPositions()
        side = _side(mol, ls)
        if _planar_bond_from_coords(ls, xyz) == ls or side is None:
            continue
        _, _, end1, end2, _, _ = ls.atoms
        origin, axis = xyz[end2], xyz[end2] - xyz[end1]
        axis /= np.linalg.norm(axis)
        for i in side:
            rel = xyz[i] - origin
            conf.SetAtomPosition(
                i, (origin + 2 * np.dot(rel, axis) * axis - rel).tolist()
            )


def _side(mol: Chem.Mol, ls: PlanarBond) -> set[int] | None:
    """The atoms on `end2`'s side of the bond or chain, or None if it is in a ring."""
    _, _, end1, end2, _, _ = ls.atoms
    nxg = nx.Graph([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    nxg.add_nodes_from(range(mol.GetNumAtoms()))
    nxg.remove_edge(ls.chain[-1] if ls.chain else end1, end2)
    side = nx.node_connected_component(nxg, end2)
    return None if end1 in side else side


# Wedges and marks


def _wedges(
    mol: Chem.Mol, lss: Sequence[Stereo], avoid: set[_Pair]
) -> dict[_Pair, bool]:
    """A wedge (True) or hash (False) for each determined `Tetrahedral` and `AtropBond`.

    Keys are (from, to): the narrow end is on `from`. Each mark is checked by lifting
    `to` out of the page (toward the viewer for a wedge) and reading the LS back. A
    bond gets at most one mark. A centre's mark goes on an H if it can, and on a bond
    to another stereo atom, or one in `avoid` (a reaction's breaking and forming
    bonds), last; an `AtropBond`'s goes on a substituent of `end2`, else of `end1`.
    """
    xyz = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    xyz[:, 2] = 0.0
    stereo_atoms: set[int] = set()
    for ls in lss:
        if isinstance(ls, PlanarBond | AtropBond):
            stereo_atoms.update(ls.atoms[2:4])
        else:
            stereo_atoms.add(ls.atoms[0])

    def rank(start: int, n: int) -> tuple[bool, bool, bool, int]:
        bond = (min(start, n), max(start, n))
        return (bond in avoid, n in stereo_atoms, not _is_h(mol, n), n)

    wedges: dict[_Pair, bool] = {}
    for ls in lss:
        if ls.parity is None or isinstance(ls, PlanarBond):
            continue
        if isinstance(ls, Tetrahedral):
            ends = [(ls.atoms[0], ls.atoms[1:])]
        elif isinstance(ls, AtropBond):
            a, b, end1, end2, c, d = ls.atoms
            ends = [(end2, (c, d)), (end1, (a, b))]
        else:
            continue
        options = [
            (start, n)
            for start, nbrs in ends
            for n in sorted(
                (n for n in nbrs if n is not None), key=partial(rank, start)
            )
            if (start, n) not in wedges and (n, start) not in wedges
        ]
        for (start, n), up in itertools.product(options, (True, False)):
            lifted = xyz.copy()
            lifted[n, 2] = _LIFT if up else -_LIFT
            if _read(ls, lifted) == ls:
                wedges[start, n] = up
                break
    return wedges


def _read(ls: Tetrahedral | AtropBond, xyz: np.ndarray) -> Stereo:
    if isinstance(ls, Tetrahedral):
        return _tetrahedral_from_coords(ls, xyz)
    return _atrop_bond_from_coords(ls, xyz)


def _set_wedges(mol: Chem.RWMol, wedges: dict[_Pair, bool]) -> None:
    """Re-add each wedge's bond from its narrow end, with its direction."""
    for (i, j), up in wedges.items():
        order = mol.GetBondBetweenAtoms(i, j).GetBondType()
        mol.RemoveBond(i, j)
        mol.AddBond(i, j, order)
        mol.GetBondBetweenAtoms(i, j).SetBondDir(
            Chem.BondDir.BEGINWEDGE if up else Chem.BondDir.BEGINDASH
        )


def _undetermined_marks(mol: Chem.RWMol, lss: Iterable[Stereo]) -> set[int]:
    """Mark each undetermined bond LS; return the centres of the undetermined atom LS.

    A double bond is crossed. RDKit doesn't cross a cumulated double bond, so a chain
    LS puts a "?" on its middle bond instead, as does an LS on a single bond.
    """
    notes: set[int] = set()
    for ls in lss:
        if ls.parity is not None:
            continue
        if not isinstance(ls, PlanarBond | AtropBond):
            notes.add(ls.atoms[0])
            continue
        bonds = _chain_bonds(ls)
        bond = mol.GetBondBetweenAtoms(*bonds[len(bonds) // 2])
        if bond.GetBondType() == Chem.BondType.DOUBLE and not ls.chain:
            bond.SetBondDir(Chem.BondDir.EITHERDOUBLE)
            bond.SetStereo(Chem.BondStereo.STEREOANY)
        else:
            bond.SetProp("bondNote", "?")
    return notes


def _is_h(mol: Chem.Mol, i: int) -> bool:
    return mol.GetAtomWithIdx(i).GetAtomicNum() == 1
