"""
Base classes and types for gating strategies.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Protocol, TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..models import TTTModel


@dataclasses.dataclass
class GatingDecision:
    """Result of a gating decision.

    Attributes:
        should_update: Boolean or scalar array indicating whether to update.
        gating_scale: Continuous scale factor [0, 1] for soft gating.
        metrics: Additional metrics for logging/debugging.
    """

    should_update: bool | jax.Array
    gating_scale: jax.Array | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.gating_scale is None:
            # Default: hard gating (0 or 1)
            self.gating_scale = jnp.asarray(self.should_update, dtype=jnp.float32)


class GatingStrategy(abc.ABC):
    """Base class for all gating strategies.

    Gating strategies decide when to apply TTT updates based on
    various signals (loss, reconstruction error, etc.).
    """

    @abc.abstractmethod
    def decide(
        self,
        loss_skip: jax.Array | float | None = None,
        ttt_loss_init: jax.Array | float | None = None,
        ttt_loss_final: jax.Array | float | None = None,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> GatingDecision:
        """Make a gating decision.

        Args:
            loss_skip: Loss from SKIP path (forward without TTT).
            ttt_loss_init: Initial TTT reconstruction loss (before update).
            ttt_loss_final: Final TTT reconstruction loss (after update).
            rng: Random key for stochastic strategies.
            **kwargs: Additional strategy-specific arguments.

        Returns:
            GatingDecision with update decision and metrics.
        """
        ...

    def update_state(self, metrics: dict[str, Any]) -> None:
        """Update internal state (e.g., EMA).

        Called after each chunk is processed. Override in stateful strategies.

        Args:
            metrics: Metrics from the current chunk.
        """
        pass

    def reset(self) -> None:
        """Reset internal state. Override in stateful strategies."""
        pass
