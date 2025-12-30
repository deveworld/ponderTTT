"""
Fixed action gating: Always SKIP or always UPDATE.
"""

from typing import Literal

import jax.numpy as jnp

from .base import GatingDecision, GatingStrategy


class FixedGating(GatingStrategy):
    """Deterministic fixed action gating.

    Always returns the same decision (SKIP or UPDATE).
    Used as baselines for comparison.

    Example:
        >>> gating = FixedGating(action="UPDATE")
        >>> decision = gating.decide()
        >>> decision.should_update  # True
    """

    def __init__(self, action: Literal["SKIP", "UPDATE"] = "SKIP"):
        """Initialize fixed gating.

        Args:
            action: "SKIP" to never update, "UPDATE" to always update.
        """
        self.action = action
        self._should_update = action == "UPDATE"

    def decide(self, **kwargs) -> GatingDecision:
        return GatingDecision(
            should_update=jnp.array(self._should_update),
            metrics={"action": self.action},
        )
