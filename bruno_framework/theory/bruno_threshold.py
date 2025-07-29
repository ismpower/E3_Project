"""
bruno_threshold.py
------------------
Detect when a cumulative fluence curve crosses a given threshold.

Functions
---------
- compute_threshold_crossings(fluence, threshold) -> list[int]
- find_first_crossing(fluence, threshold) -> int | None
"""

import numpy as np
from typing import List, Optional, Sequence

def compute_threshold_crossings(
    fluence: Sequence[float],
    threshold: float
) -> List[int]:
    """
    Identify all indices in a fluence series where the value meets or exceeds a threshold.

    Args:
        fluence:   1D sequence of cumulative fluence values (J/m²), ordered by time.
        threshold: Fluence threshold to test against (J/m²).

    Returns:
        List of integer indices i such that fluence[i] >= threshold.
    """
    f = np.asarray(fluence, dtype=float)
    return list(np.where(f >= threshold)[0])


def find_first_crossing(
    fluence: Sequence[float],
    threshold: float
) -> Optional[int]:
    """
    Find the first index where cumulative fluence crosses the threshold.

    Args:
        fluence:   1D sequence of cumulative fluence values (J/m²).
        threshold: Fluence threshold to test against (J/m²).

    Returns:
        The first index i where fluence[i] >= threshold, or None if never crossed.
    """
    crossings = compute_threshold_crossings(fluence, threshold)
    return crossings[0] if crossings else None
