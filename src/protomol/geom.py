"""Geometry data structure and functions."""

from collections.abc import Sequence
from typing import Annotated

import numpy
import pint
import py3Dmol
import pyparsing as pp
from numpy.typing import ArrayLike
from pydantic import BaseModel, BeforeValidator
from pyparsing import pyparsing_common as ppc

from . import rd
from .util import units
from .util.types import NDArray


class Geometry(BaseModel):
    """Geometry data structure.

    :param symbols: Atomic symbols
    :param coordinates: Atomic coordinates (units: angstrom)
    :param charge: Total molecular charge
    :param spin: Total electronic spin (multiplicity - 1)
    """

    symbols: Annotated[list[str], BeforeValidator(list)]
    coordinates: NDArray
    charge: int = 0
    spin: int = 0


# constructors
def from_data(
    symbols: Sequence[str],
    coordinates: ArrayLike,
    charge: int = 0,
    spin: int = 0,
    unit: str = units.DISTANCE_UNIT,
) -> Geometry:
    """Generate geometry from data.

    :param symbols: Atomic symbols
    :param coordinates: Atomic coordinates
    :param charge: Total molecular charge
    :param spin: Total electronic spin (multiplicity - 1)
    :param unit: Distance unit
    :return: Geometry
    """
    return Geometry(
        symbols=list(symbols),
        coordinates=numpy.array(
            pint.Quantity(coordinates, unit).m_as(units.DISTANCE_UNIT)
        ),
        charge=charge,
        spin=spin,
    )


def from_rdkit_molecule(mol: rd.mol.Mol) -> Geometry:
    """Generate geometry from RDKit molecule.

    :param mol: Molecule
    :return: Geometry
    """
    mol = rd.mol.with_coordinates(mol)
    coords = rd.mol.coordinates(mol)
    assert coords is not None, f"RDKit molecule should have coordinates: {mol}"
    return Geometry(
        symbols=rd.mol.symbols(mol),
        coordinates=coords,
        charge=rd.mol.charge(mol),
        spin=rd.mol.spin(mol),
    )


# core properties
def count(geo: Geometry) -> int:
    """Get number of atoms.

    :param geo: Geometry
    :return: Number of atoms
    """
    return len(symbols(geo))


def symbols(geo: Geometry) -> list[str]:
    """Get atomic symbols.

    :param geo: Geometry
    :return: Symbols
    """
    return geo.symbols


def coordinates(geo: Geometry, unit: str = units.DISTANCE_UNIT) -> NDArray:
    """Get atomic coordinates.

    :param geo: Geometry
    :param unit: Distance unit
    :return: Coordinates
    """
    return geo.coordinates * units.distance_conversion(units.DISTANCE_UNIT, unit)


def charge(geo: Geometry) -> int:
    """Get total molecular charge.

    :param geo: Geometry
    :return: Charge
    """
    return geo.charge


def spin(geo: Geometry) -> int:
    """Get total electronic spin.

    spin = number of unpaired electrons = multiplicity - 1

    :param geo: Geometry
    :return: Spin
    """
    return geo.spin


# calculated properties
def multiplicity(geo: Geometry) -> int:
    """Get spin multiplicity.

    :param geo: Geometry
    :return: Spin multiplicity
    """
    return spin(geo) + 1


# conversions
def xyz_string(geo: Geometry) -> str:
    """Serialize as XYZ string.

    :param geo: Geometry
    :return: XYZ string
    """
    natms = count(geo)
    symbs = symbols(geo)
    xyzs = coordinates(geo, unit="angstrom")
    return f"{natms}\n\n" + "\n".join(
        f"{s} {x:10.6f} {y:10.6f} {z:10.6f}"
        for s, (x, y, z) in zip(symbs, xyzs, strict=True)
    )


CHAR = pp.Char(pp.alphas)
SYMBOL = pp.Combine(CHAR + pp.Opt(CHAR))
XYZ_LINE = SYMBOL + pp.Group(ppc.fnumber * 3) + pp.Suppress(... + pp.LineEnd())


def from_xyz_string(geo_str: str) -> Geometry:
    """Read geometry from XYZ string.

    :param geo_str: _description_
    :return: _description_
    """
    geo_str = geo_str.strip()
    lines = geo_str.splitlines()[2:]
    if not lines:
        return from_data(symbols=[], coordinates=[])

    symbs, coords = zip(*[XYZ_LINE.parse_string(line).as_list() for line in lines])
    geo = from_data(symbols=symbs, coordinates=coords, unit="angstrom")
    return geo


def display(geo: Geometry, width: int = 600, height: int = 450) -> None:
    """Display geometry.

    :param geo: Geometry
    :param width: Width
    :param height: Height
    """
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz_string(geo), "xyz")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    viewer.zoomTo()
    viewer.show()
