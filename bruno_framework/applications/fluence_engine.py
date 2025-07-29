"""
fluence_engine.py
-----------------
Compute cumulative fluence curves from time‐series flux data.

Fluence is defined as the time‐integral of flux (J/m²/s) divided by 4πD²,
where D is the source distance in meters.

Functions
---------
- integrate_fluence(times, flux) -> float
- compute_fluence_curve(times, flux, distance_m) -> pandas.DataFrame
"""

import numpy as np
import pandas as pd
from typing import Sequence, Union

def integrate_fluence(
    times: Sequence[float],
    flux: Sequence[float]
) -> float:
    """
    Numerically integrate a flux time series to get total fluence (energy/area).

    Args:
        times: 1D sequence of time points (in seconds), not necessarily sorted.
        flux:  1D sequence of flux values (in J/m²/s), same length as `times`.

    Returns:
        Total fluence (in J/m²), i.e. ∫ flux dt.
    """
    # Convert to numpy arrays
    t = np.asarray(times, dtype=float)
    f = np.asarray(flux, dtype=float)

    # Sort by time
    idx = np.argsort(t)
    t = t[idx]
    f = f[idx]

    # Compute time differences
    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0.0

    # Integrate flux over time
    return float(np.sum(f * dt))


def compute_fluence_curve(
    times: Sequence[float],
    flux: Sequence[float],
    distance_m: float
) -> pd.DataFrame:
    """
    Compute a cumulative fluence curve for plotting or thresholding.

    Args:
        times:      1D sequence of time points (s), not necessarily sorted.
        flux:       1D sequence of flux values (J/m²/s), same length as `times`.
        distance_m: Distance to source (m).

    Returns:
        DataFrame with columns:
          - 'time':    sorted time points (s)
          - 'fluence': cumulative fluence at each time (J/m²)
    """
    t = np.asarray(times, dtype=float)
    f = np.asarray(flux, dtype=float)
    idx = np.argsort(t)
    t = t[idx]
    f = f[idx]

    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0.0
    cum_energy = np.cumsum(f * dt)            # total energy per area
    fluence = cum_energy / (4.0 * np.pi * distance_m**2)

    return pd.DataFrame({"time": t, "fluence": fluence})
