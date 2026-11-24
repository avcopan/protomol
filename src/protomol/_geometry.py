"""`from_geometry`: between graphs and autogeom geometries (P4).

Atom i of the graph is atom i of the geometry. The LS sites are those `from_smiles`
gives (the low-T approximation, through the same pipeline), and each site's parity is
read from the coordinates.
"""

import numpy as np
from autogeom import Geometry, adjacency_matrix, rd
from autogeom.exc import GeometryConversionError

from ._smiles import _from_rdkit_mol
from ._stereo import AtropBond, PlanarBond, Stereo, Tetrahedral
from ._types import GraphError, StereoMolGraph, with_stereo
from ._vendor.stereomolgraph.coords import handedness

type _Coords = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type _Vector = np.ndarray[tuple[int], np.dtype[np.float64]]

# Below these, a centre is flat or a bond twisted by about 90° (no parity is read)
_MIN_VOLUME = 0.3  # |det| of three unit-vector differences; ideal tetrahedron: 3.08
_MIN_COSINE = 0.3  # |cos| (planar) or |sin| (axial) of the dihedral between the ends


def from_geometry(geo: Geometry) -> StereoMolGraph:
    """The stereo graph of a geometry, keeping its atom order.

    Bonds come from covalent radii (`autogeom.adjacency_matrix`, with valences capped),
    and a Lewis structure from autogeom's solver, with the geometry's charge and spin.
    The LS sites are the ones `from_smiles` would give for that structure (the low-T
    approximation), and every site's parity is read from the coordinates, trivial sites
    included, so the graph describes this geometry atom for atom. A site whose
    coordinates don't decide it (a flat centre, a double bond twisted by about 90°)
    stays undetermined.

    >>> import autogeom
    >>> geo = autogeom.Geometry(
    ...     symbols=["O", "H", "H"],
    ...     coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    ...     charge=0,
    ...     spin=0,
    ... )
    >>> from_geometry(geo).bonds
    ((0, 1), (0, 2))

    Raises:
        GraphError: If no Lewis structure fits the bonds, charge and spin.
    """
    amat = adjacency_matrix(geo, enforce_valence=True)
    bonds = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(amat)), strict=True)]
    try:
        mol = rd.from_connectivity(geo.symbols, bonds, charge=geo.charge, spin=geo.spin)
    except GeometryConversionError as err:
        raise GraphError(str(err)) from err
    gra = _from_rdkit_mol(mol, charge=geo.charge, spin=geo.spin)
    return _with_parities_from(gra, np.asarray(geo.coordinates, dtype=np.float64))


def _with_parities_from(gra: StereoMolGraph, xyz: _Coords) -> StereoMolGraph:
    """`gra` with each LS's parity read from the coordinates (undetermined if unclear).

    Only the kinds the low-T scheme gives: `Tetrahedral`, `PlanarBond`, `AtropBond`.
    """
    lss = [_ls_from_coords(ls, xyz) for ls in (*gra.atom_stereo, *gra.bond_stereo)]
    return with_stereo(gra, *lss)


def _ls_from_coords(ls: Stereo, xyz: _Coords) -> Stereo:
    """The configuration of `ls` in the coordinates, or `ls` undetermined if unclear."""
    if isinstance(ls, Tetrahedral):
        return _tetrahedral_from_coords(ls, xyz)
    if isinstance(ls, PlanarBond):
        return _planar_bond_from_coords(ls, xyz)
    if isinstance(ls, AtropBond):
        return _atrop_bond_from_coords(ls, xyz)
    msg = f"Reading a {ls.kind} from coordinates is not implemented"
    raise NotImplementedError(msg)


def _tetrahedral_from_coords(ls: Tetrahedral, xyz: _Coords) -> Tetrahedral:
    """Stereomolgraph's `handedness` of n1…n4 as unit vectors from the centre.

    A lone pair (None) points opposite the sum of the other three.
    """
    center, *nbrs = ls.atoms
    vecs = [None if n is None else _unit(xyz[n] - xyz[center]) for n in nbrs]
    if sum(v is None for v in vecs) > 1:
        return Tetrahedral(atoms=ls.atoms)
    known = [v for v in vecs if v is not None]
    points = np.array([-np.sum(known, axis=0) if v is None else v for v in vecs])
    if abs(np.linalg.det(points[1:] - points[0])) < _MIN_VOLUME:
        return Tetrahedral(atoms=ls.atoms)
    parity = int(handedness(points))
    return Tetrahedral(atoms=ls.atoms, parity=1 if parity > 0 else -1)


def _planar_bond_from_coords(ls: PlanarBond, xyz: _Coords) -> PlanarBond:
    """A cis to c (the atoms as given) or trans (c and d swapped)."""
    sides = _end_sides(ls.atoms, xyz)
    if sides is None:
        return PlanarBond(atoms=ls.atoms, chain=ls.chain)
    s1, s2, _ = sides
    cosine = float(np.dot(s1, s2))
    if abs(cosine) < _MIN_COSINE:
        return PlanarBond(atoms=ls.atoms, chain=ls.chain)
    a, b, end1, end2, c, d = ls.atoms
    atoms = (a, b, end1, end2, c, d) if cosine > 0 else (a, b, end1, end2, d, c)
    return PlanarBond(atoms=atoms, parity=0, chain=ls.chain)


def _atrop_bond_from_coords(ls: AtropBond, xyz: _Coords) -> AtropBond:
    """Parity 1 when the triple product n.(s1 x s2) > 0, n pointing from end1 to end2.

    s1 points from b to a and s2 from d to c, both perpendicular to the axis. This reads
    stereomolgraph's `AtropBond` diagram with ◀ as a wedge toward the viewer: with c, d
    in the page, parity 1 has a in front.
    """
    sides = _end_sides(ls.atoms, xyz)
    if sides is None:
        return AtropBond(atoms=ls.atoms, chain=ls.chain)
    s1, s2, axis = sides
    sine = float(np.dot(axis, np.cross(s1, s2)))
    if abs(sine) < _MIN_COSINE:
        return AtropBond(atoms=ls.atoms, chain=ls.chain)
    return AtropBond(atoms=ls.atoms, parity=1 if sine > 0 else -1, chain=ls.chain)


def _end_sides(
    atoms: tuple[int | None, ...], xyz: _Coords
) -> tuple[_Vector, _Vector, _Vector] | None:
    """For a bond LS (a, b, end1, end2, c, d): (s1, s2, axis) as unit vectors.

    s1 is the direction from b to a and s2 from d to c, perpendicular to the end1 → end2
    axis; a missing substituent (None) counts as the zero vector. None if either end
    has no direction (its substituents lie on the axis).
    """
    a, b, end1, end2, c, d = atoms
    assert end1 is not None  # noqa: S101  (bond LS ends are atoms)
    assert end2 is not None  # noqa: S101
    axis = _unit(xyz[end2] - xyz[end1])

    def side(x: int | None, y: int | None, end: int) -> _Vector | None:
        vec = sum(
            (
                sign * _unit(xyz[n] - xyz[end])
                for sign, n in ((1, x), (-1, y))
                if n is not None
            ),
            start=np.zeros(3),
        )
        vec = vec - np.dot(vec, axis) * axis
        norm = np.linalg.norm(vec)
        return None if norm < 1e-3 else vec / norm  # noqa: PLR2004

    s1, s2 = side(a, b, end1), side(c, d, end2)
    if s1 is None or s2 is None:
        return None
    return s1, s2, axis


def _unit(vec: _Vector) -> _Vector:
    return vec / np.linalg.norm(vec)
