"""Individual RDKit molecule."""

import copy

import numpy
import py3Dmol
from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Mol, rdDistGeom, rdmolfiles

from ..util import units
from ..util.types import NDArray

RDKIT_DISTANCE_UNIT = "angstrom"


def from_smiles(smi: str, with_coords: bool = False) -> Mol:
    """Generate an RDKit molecule from SMILES.

    :param smi: SMILES string
    :param with_coords: Whether to add coordinates
    :return: RDKit molecule
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if with_coords:
        mol = with_coordinates(mol)
    return mol


# properties
def symbols(mol: Mol) -> list[str]:
    """Get atomic symbols.

    :param mol: RDKit molecule
    :return: Symbols
    """
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def coordinates(mol: Mol, unit: str = units.DISTANCE_UNIT) -> NDArray | None:
    """Get atomic coordinates.

    Requires an embedded molecule (otherwise, returns None).

    :param mol: RDKit molecule
    :return: Coordinates
    """
    if not has_coordinates(mol):
        return None

    natms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    coords = [conf.GetAtomPosition(i) for i in range(natms)]
    coords = numpy.array(coords, dtype=numpy.float64)
    return coords * units.distance_conversion(RDKIT_DISTANCE_UNIT, unit)


def charge(mol: Mol) -> int:
    """Get molecular charge.

    :param mol: RDKit molecule
    :return: Charge
    """
    return Chem.GetFormalCharge(mol)


def spin(mol: Mol) -> int:
    """Determine (or guess) molecular spin.

    spin = number of unpaired electrons = multiplicity - 1

    TODO: Add flags to decide between high- and low-spin guess where ambiguous.

    :param mol: RDKit molecule
    :return: Spin
    """
    return Descriptors.NumRadicalElectrons(mol)


# boolean properties
def has_coordinates(mol: Mol) -> bool:
    """Determine if RDKit molecule has coordinates.

    :param mol: RDKit molecule
    :return: `True` if it does, `False` if not
    """
    return bool(mol.GetNumConformers())


# convert
def image(
    mol: Mol, *, label: bool = True, num_dct: dict[int, int] | None = None
) -> Image:
    """Generate a display-able image.

    If label=True but no mapping is specified, the flat indices will be used.

    :param mols: RDKit molecules
    :param label: Whether to label the atoms
    :param mapping: An alternative mapping
    :return: PIL Image
    """
    if label or num_dct is not None:
        mol = with_numbers(mol, num_dct=num_dct, in_place=False)

    return Draw.MolToImage(mol)


def view(
    mol: Mol, *, label: bool = True, width: int = 600, height: int = 450
) -> py3Dmol.view:
    """View molecule as a 3D structure.

    :param geo: Geometry
    :param width: Width
    :param height: Height
    """
    xyz_str = Chem.MolToXYZBlock(mol)

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(xyz_str, "xyz")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.3}})

    if label:
        for idx in range(mol.GetNumAtoms()):
            viewer.addLabel(
                idx,
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"index": idx},
            )

    viewer.zoomTo()
    return viewer


# transformations
def with_numbers(
    mol: Mol, num_dct: dict[int, int] | None = None, in_place: bool = False
) -> Mol:
    """Add atom numbers to RDKit molecule.

    If no numbers dictionary is specified, the atom indices will be used.

    :param mol: RDKit molecule
    :param num_dct: Alternative numbers to use, by atom index
    :param in_place: Whether to modify the molecule in place
    :return: RDKit molecule
    """
    mol = mol if in_place else copy.deepcopy(mol)
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        num = atom_idx if num_dct is None else num_dct[atom_idx]
        atom.SetProp("molAtomMapNumber", str(num))
        # # This doesn't work because a value of 0 clears the property:
        # atom.SetAtomMapNum(num)
    return mol


def with_coordinates(mol: Mol, in_place: bool = False) -> Mol:
    """Add coordinates to RDKit molecule, if missing.

    :param mol: RDKit molecule
    :param in_place: Whether to modify the molecule in place
    :return: RDKit molecule
    """
    if not has_coordinates(mol):
        mol = mol if in_place else copy.deepcopy(mol)
        rdDistGeom.EmbedMolecule(mol)
    return mol


def xyz_string(mol: Mol) -> str:
    """Generate an XYZ string from an RDKit molecule.

    :param mol: RDKit molecule
    :return: XYZ string
    """
    return rdmolfiles.MolToXYZBlock(mol)
