"""Unit functions."""

from pint import Quantity

DISTANCE_UNIT = "angstrom"  # The internal distance unit for this code


def distance_conversion(unit0: str = DISTANCE_UNIT, unit: str = DISTANCE_UNIT) -> float:
    """Determine conversion between distance units.

    :param unit0: Initial distance unit
    :param unit: Final distance unit
    :return: Conversion factor
    """
    if unit0 == unit:
        return 1.0

    return Quantity(unit0).m_as(unit)
