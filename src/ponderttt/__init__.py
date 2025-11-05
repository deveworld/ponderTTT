"""
PonderTTT: Learning to Ponder with Variable-Depth Test-Time Training

Adaptive inner-loop iteration mechanism for Test-Time Training layers.
"""

__version__ = "0.1.0"

from .models.adaptive_ttt import AdaptiveTTT, HeuristicAdaptiveTTT
from .models.ttt_linear import TTTLinear

__all__ = [
    "TTTLinear",
    "AdaptiveTTT",
    "HeuristicAdaptiveTTT",
]
