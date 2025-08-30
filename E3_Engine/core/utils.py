"""
utils.py
--------
General helper routines: config loading, path resolution, unit conversion.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file into a dictionary.

    Args:
        path: Path to a .json config file.

    Returns:
        A dict representing the JSON contents.
    """
    with open(path, 'r') as f:
        return json.load(f)


def resolve_path(*parts: str) -> str:
    """
    Safely build a filesystem path from segments, expanding ~ and constants.

    Args:
        *parts: Path components, e.g. ('data', 'raw', 'file.csv').

    Returns:
        The absolute, normalized path as a string.
    """
    return str(Path(*parts).expanduser().resolve())


def unit_convert(
    value: float,
    from_unit: str,
    to_unit: str,
    conversion_factors: Dict[str, float] = None
) -> float:
    """
    Convert a scalar between simple units via a lookup table.

    Args:
        value: Numeric value to convert.
        from_unit: Unit of the input value (e.g. 'km', 'm').
        to_unit: Desired unit (e.g. 'm', 'cm').
        conversion_factors:
            Optional mapping like {'km': 1e3, 'm': 1.0, 'cm': 1e-2}.
            If None, defaults to {'m':1, 'km':1e-3, 'cm':1e2}.

    Returns:
        The value in the target units.

    Raises:
        KeyError if `from_unit` or `to_unit` not in the factor map.
    """
    if conversion_factors is None:
        conversion_factors = {'m': 1.0, 'km': 1e-3, 'cm': 1e2}

    # Convert to base unit, then to target
    base = value / conversion_factors[from_unit]
    return base * conversion_factors[to_unit]
