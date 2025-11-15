"""
PonderTTT: Learning When to Ponder During Test-Time Training

JAX/Flax implementation for TPU optimization.
"""

__version__ = "0.2.0"

from . import data, models, training, utils

__all__ = ["data", "models", "training", "utils"]
