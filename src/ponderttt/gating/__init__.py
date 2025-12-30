"""
Gating Module for PonderTTT.

This module provides unified gating strategies for deciding when to apply
Test-Time Training (TTT) updates.

Available Strategies:
    - FixedGating: Always SKIP or always UPDATE
    - RandomGating: Update with probability p
    - ThresholdGating: Update based on loss threshold
    - EMAGating: Adaptive threshold with EMA adjustment
    - ReconstructionGating: TTT reconstruction loss based gating
"""

from .base import GatingDecision, GatingStrategy
from .fixed import FixedGating
from .random import RandomGating
from .threshold import ThresholdGating
from .ema import EMAGating
from .reconstruction import ReconstructionGating

__all__ = [
    "GatingDecision",
    "GatingStrategy",
    "FixedGating",
    "RandomGating",
    "ThresholdGating",
    "EMAGating",
    "ReconstructionGating",
]
