"""
Experiment scripts for PonderTTT.
"""

from .config import (
    ExperimentConfig,
    ExperimentModelConfig,
    TrainingConfig,
    get_large_config,
    get_small_config,
    get_medium_config,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentModelConfig",
    "TrainingConfig",
    "get_small_config",
    "get_medium_config",
    "get_large_config",
]
