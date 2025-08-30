# Bruno Engine Core Package
# --------------------------
# Modules:
# - fluence_engine: entropy fluence simulation
# - bruno_threshold: trigger crossing logic
# - registry_logger: registry entry creator
# - utils: shared helpers
# - entropy_first_model: energy-temperature relations

"""
`bruno_core` package initializer. Exposes key functions from submodules for easy import.
"""

__version__ = "0.1.0"

# Core functions
from .fluence_engine import integrate_fluence, compute_fluence_curve
from .bruno_threshold import compute_threshold_crossings, find_first_crossing
from .registry_logger import log_event
from .utils import load_config, resolve_path, unit_convert
from .entropy_first_model import entropy_field_model, temperature_from_entropy, compare_model_to_data

# Public API
__all__ = [
    "integrate_fluence",
    "compute_fluence_curve",
    "compute_threshold_crossings",
    "find_first_crossing",
    "log_event",
    "load_config",
    "resolve_path",
    "unit_convert",
    "entropy_field_model",
    "temperature_from_entropy",
    "compare_model_to_data",
]
