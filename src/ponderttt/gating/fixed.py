"""
Fixed action gating: Always SKIP or always UPDATE.
"""

from typing import Literal

import jax
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

    def decide(
        self,
        loss_skip: jax.Array | float | None = None,
        ttt_loss_init: jax.Array | float | None = None,
        ttt_loss_final: jax.Array | float | None = None,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> GatingDecision:
        return GatingDecision(
            should_update=jnp.array(self._should_update),
            metrics={"action": self.action},
        )
