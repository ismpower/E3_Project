"""
entropy_first_model.py
----------------------
Module implementing the entropy-first framework to derive energy-temperature relations.

Assumptions:
1. Entropy S(E) describes the count of distinguishable energy field configurations.
2. Temperature T(E) emerges as the reciprocal derivative of entropy (1/T = dS/dE).

Functions
---------
- entropy_field_model(E, alpha, E0) -> Union[float, np.ndarray]
- temperature_from_entropy(E, alpha, E0) -> Union[float, np.ndarray]
- compare_model_to_data(data, energy_col, temperature_col, alpha, E0) -> pd.DataFrame
"""

from typing import Union
import numpy as np
import pandas as pd

# Constants
k_B: float = 1.380649e-23  # Boltzmann constant (J/K)
c: float   = 299792458     # Speed of light (m/s)
h: float   = 6.62607015e-34  # Planck constant (J*s)


def entropy_field_model(
    E: Union[float, np.ndarray],
    alpha: float = 1.5,
    E0: float = 1e-21
) -> Union[float, np.ndarray]:
    """
    Compute entropy as a power-law function of energy.

    S(E) = k_B * (E / E0)**alpha

    Args:
        E: scalar or array of energy values (J).
        alpha: exponent for power law growth of entropy.
        E0: reference energy scale (J).

    Returns:
        Entropy in J/K, same shape as input E.
    """
    return k_B * np.power(E / E0, alpha)


def temperature_from_entropy(
    E: Union[float, np.ndarray],
    alpha: float = 1.5,
    E0: float = 1e-21
) -> Union[float, np.ndarray]:
    """
    Compute temperature as the reciprocal derivative of entropy:
    T(E) = 1 / (dS/dE), where S(E) = k_B * (E/E0)**alpha

    Args:
        E: scalar or array of energy values (J).
        alpha: exponent for entropy model.
        E0: reference energy scale (J).

    Returns:
        Temperature in Kelvin (K), same shape as input E.
    """
    # dS/dE = alpha * k_B / E0 * (E/E0)**(alpha-1)
    dSdE = alpha * k_B / E0 * np.power(E / E0, alpha - 1)
    return 1.0 / dSdE


def compare_model_to_data(
    data: pd.DataFrame,
    energy_col: str,
    temperature_col: str,
    alpha: float = 1.5,
    E0: float = 1e-21
) -> pd.DataFrame:
    """
    Fit the temperature model to observational data and compute residuals.

    Args:
        data: DataFrame with observational columns.
        energy_col: column name for energy values.
        temperature_col: column name for observed temperatures.
        alpha: exponent for entropy model.
        E0: reference energy scale.

    Returns:
        DataFrame with additional columns:
          - 'model_temperature': predicted T(E)
          - 'residual': observed minus model temperature
    """
    df = data.copy()
    E = df[energy_col].values
    model_T = temperature_from_entropy(E, alpha=alpha, E0=E0)
    df['model_temperature'] = model_T
    df['residual'] = df[temperature_col] - df['model_temperature']
    return df
