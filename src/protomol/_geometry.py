"""`from_geometry` and `embed`: between graphs and autogeom geometries (P4).

Atom i of the graph is atom i of the geometry. `from_geometry` takes the LS sites
`from_smiles` gives (the low-T approximation, through the same pipeline) and reads each
site's parity from the coordinates. `embed` runs RDKit's ETKDG on the graph's Lewis
structure, with its determined LS as constraints, and checks the result with the same
parity readers.
"""

import numpy as np
from autogeom import Geometry, adjacency_matrix, rd
from autogeom.exc import GeometryConversionError
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from ._lewis import _lewis
from ._smiles import _from_rdkit_mol
from ._stereo import AtropBond, PlanarBond, Stereo, Tetrahedral
from ._types import GraphError, MolGraph, StereoMolGraph, with_stereo
from ._vendor.stereomolgraph.coords import handedness

type _Coords = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type _Vector = np.ndarray[tuple[int], np.dtype[np.float64]]

# Below these, a centre is flat or a bond twisted by about 90° (no parity is read)
_MIN_VOLUME = 0.3  # |det| of three unit-vector differences; ideal tetrahedron: 3.08
_MIN_COSINE = 0.3  # |cos| (planar) or |sin| (axial) of the dihedral between the ends

_EMBED_ATTEMPTS = 100  # ETKDG random seeds 0, 1, …
_PYRAMID_HEIGHT = 0.35  # Å of a pyramidal centre above its neighbours' plane
_FRAGMENT_GAP = 3.0  # Å between the fragments of a disconnected graph, along x
_FLIP_TAG = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


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


def embed(gra: MolGraph | StereoMolGraph) -> Geometry:
    """A 3D geometry of the graph, in its atom order, with its charge and spin.

    RDKit's ETKDG embeds a Lewis structure of the graph (`_lewis`), with each determined
    `Tetrahedral` (lone-pair centres included) as a chiral tag and each determined
    `PlanarBond` on a double bond as cis/trans bond stereo. Every determined LS is then
    read back from the coordinates, so `from_geometry(embed(gra))` has the same LS. An
    LS ETKDG can't hold (a radical's planar bond, a chain LS) is met by embedding again
    with the next random seed. Undetermined LS take whatever configuration comes out.
    The seeds are fixed, so the result is reproducible. The fragments of a disconnected
    graph are set 3 Å apart.

    >>> from protomol import examples, from_geometry
    >>> gra = examples.species()
    >>> geo = embed(gra)
    >>> geo.symbols == list(gra.symbols), geo.coordinates.shape
    (True, (16, 3))
    >>> from_geometry(geo) == gra
    True

    Raises:
        TypeError: For a reaction graph.
        GraphError: If no Lewis structure fits the graph, or no embedding meets its LS.
    """
    if not isinstance(gra, MolGraph | StereoMolGraph):
        msg = f"embed takes a MolGraph or a StereoMolGraph, not {type(gra).__name__}"
        raise TypeError(msg)
    lss = (
        [ls for ls in (*gra.atom_stereo, *gra.bond_stereo) if ls.parity is not None]
        if isinstance(gra, StereoMolGraph)
        else []
    )
    mol = Chem.RWMol(_lewis(gra))
    tags = _set_stereo_tags(mol, lss)
    for seed in range(_EMBED_ATTEMPTS):
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = seed
        if rdDistGeom.EmbedMolecule(mol, params) != 0:
            continue
        xyz = _pyramidalized(mol.GetConformer().GetPositions(), lss)
        wrong = [ls for ls in lss if _ls_from_coords(ls, xyz) != ls]
        if not wrong:
            coords = _spread_fragments(mol, xyz)
            return Geometry(
                symbols=list(gra.symbols),
                coordinates=coords,
                charge=gra.charge,
                spin=gra.spin,
            )
        # A tag whose centre came out inverted: RDKit's sense is the other one
        for ls in wrong:
            center = ls.atoms[0]
            if isinstance(ls, Tetrahedral) and center in tags:
                tags[center] = _FLIP_TAG[tags[center]]
                mol.GetAtomWithIdx(center).SetChiralTag(tags[center])
    msg = f"No embedding in {_EMBED_ATTEMPTS} attempts meets the LS of {gra}"
    raise GraphError(msg)


def _set_stereo_tags(mol: Chem.RWMol, lss: list[Stereo]) -> dict[int, Chem.ChiralType]:
    """Tag `mol` for ETKDG: chiral tags on the centres, cis/trans on double bonds.

    A chiral tag's sense depends on RDKit's neighbour order, so every centre starts
    clockwise and `embed` flips the ones that come out inverted. Returns the tags.
    """
    tags = {}
    for ls in lss:
        if isinstance(ls, Tetrahedral):
            tags[ls.atoms[0]] = Chem.ChiralType.CHI_TETRAHEDRAL_CW
            mol.GetAtomWithIdx(ls.atoms[0]).SetChiralTag(tags[ls.atoms[0]])
        elif isinstance(ls, PlanarBond) and not ls.chain:
            a, b, end1, end2, c, d = ls.atoms
            bond = mol.GetBondBetweenAtoms(end1, end2)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            # a is cis to c; with b or d in their place, the sense flips
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
    return tags


def _pyramidalized(xyz: _Coords, lss: list[Stereo]) -> _Coords:
    """`xyz` with each flat centre of a determined three-neighbour LS made pyramidal.

    ETKDG keeps an atom flat if RDKit calls it sp2, whatever its chiral tag: the N of an
    N-acyl aziridine, or a radical centre. Each `Tetrahedral` with a `None` slot (a lone
    pair or an unpaired electron) whose centre the coordinates leave undetermined is
    moved `_PYRAMID_HEIGHT` out of its neighbours' plane, to the side of its parity.
    """
    xyz = np.array(xyz, dtype=np.float64)
    for ls in lss:
        if (
            not isinstance(ls, Tetrahedral)
            or None not in ls.atoms
            or _tetrahedral_from_coords(ls, xyz).parity is not None
        ):
            continue
        center = ls.atoms[0]
        nbrs = [n for n in ls.atoms[1:] if n is not None]
        normal = _unit(
            np.cross(xyz[nbrs[1]] - xyz[nbrs[0]], xyz[nbrs[2]] - xyz[nbrs[0]])
        )
        foot = xyz[center] - np.dot(xyz[center] - xyz[nbrs[0]], normal) * normal
        for sign in (1, -1):
            xyz[center] = foot + sign * _PYRAMID_HEIGHT * normal
            if _tetrahedral_from_coords(ls, xyz) == ls:
                break
    return xyz


def _spread_fragments(mol: Chem.Mol, xyz: _Coords) -> _Coords:
    """Each fragment centred, then set along x with `_FRAGMENT_GAP` between them."""
    coords = np.array(xyz, dtype=np.float64)
    start = 0.0
    for frag in Chem.GetMolFrags(mol):
        idx = list(frag)
        part = coords[idx] - coords[idx].mean(axis=0)
        part[:, 0] += start - part[:, 0].min()
        coords[idx] = part
        start = part[:, 0].max() + _FRAGMENT_GAP
    return coords


def _with_parities_from_geometry(gra: StereoMolGraph, geo: Geometry) -> StereoMolGraph:
    """`gra` with each undetermined LS's parity read from `geo`, atom i on atom i.

    How a scheme's added sites get their parities (P57): `smg_low_t@1` applies it to
    `low_t(gra)` (P53). It uses `from_geometry`'s readers, and a site whose coordinates
    don't decide it stays undetermined.

    Raises:
        GraphError: If the geometry's symbols differ from the graph's.
    """
    if tuple(geo.symbols) != gra.symbols:
        msg = f"The geometry's atoms {geo.symbols} differ from the graph's"
        raise GraphError(msg)
    xyz = np.asarray(geo.coordinates, dtype=np.float64)
    lss = [
        _ls_from_coords(ls, xyz)
        for ls in (*gra.atom_stereo, *gra.bond_stereo)
        if ls.parity is None
    ]
    return with_stereo(gra, *lss)


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
