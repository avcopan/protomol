"""`_lewis(gra)`: a perceived Lewis structure, as an RDKit molecule (P43).

A graph carries no bond orders; code that needs a Lewis structure (`embed` now,
`display` later) gets one here. The interim method is autogeom's integer-program solver
(`autogeom.rd.from_connectivity`), which takes the charge and spin into account, so
radicals work; stereomol M2.4 replaces it.
"""

from autogeom import rd
from autogeom.exc import GeometryConversionError
from rdkit import Chem

from ._types import GraphError, MolGraph, StereoMolGraph


def _lewis(gra: MolGraph | StereoMolGraph) -> Chem.Mol:
    """A Lewis structure of `gra`: an RDKit molecule in its atom order, H explicit.

    >>> from protomol import from_smiles
    >>> mol = _lewis(from_smiles("C=C[CH2]"))  # allyl: one of its two structures
    >>> sorted(b.GetBondTypeAsDouble() for b in mol.GetBonds())[-2:]
    [1.0, 2.0]
    >>> sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    1

    Raises:
        GraphError: If no Lewis structure fits the bonds, charge and spin.
    """
    try:
        return rd.from_connectivity(
            gra.symbols, gra.bonds, charge=gra.charge, spin=gra.spin
        )
    except GeometryConversionError as err:
        raise GraphError(str(err)) from err
