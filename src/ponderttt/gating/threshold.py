"""
Threshold-based gating: Update if signal exceeds threshold.
"""

from typing import Literal

import jax
import jax.numpy as jnp

from .base import GatingDecision, GatingStrategy


class ThresholdGating(GatingStrategy):
    """Threshold-based gating.

    Updates when a signal (loss, reconstruction error) exceeds
    a fixed threshold.

    Signals:
        - "loss_skip": Forward loss without TTT (high = hard chunk)
        - "ttt_loss_init": Initial reconstruction loss (high = needs update)
        - "ttt_improvement": Improvement from TTT (high = beneficial update)

    Example:
        >>> gating = ThresholdGating(threshold=2.0, signal="loss_skip")
        >>> decision = gating.decide(loss_skip=2.5)
        >>> decision.should_update  # True (2.5 > 2.0)
    """

    def __init__(
        self,
        threshold: float = 2.0,
        signal: Literal[
            "loss_skip", "ttt_loss_init", "ttt_improvement"
        ] = "ttt_loss_init",
    ):
        """Initialize threshold gating.

        Args:
            threshold: Decision threshold.
            signal: Which signal to use for gating decision.
        """
        self.threshold = threshold
        self.signal = signal

    def decide(
        self,
        loss_skip: jax.Array | float | None = None,
        ttt_loss_init: jax.Array | float | None = None,
        ttt_loss_final: jax.Array | float | None = None,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> GatingDecision:
        # Select signal based on configuration
        if self.signal == "loss_skip":
            if loss_skip is None:
                raise ValueError("loss_skip required for signal='loss_skip'")
            signal_value = loss_skip
        elif self.signal == "ttt_loss_init":
            if ttt_loss_init is None:
                raise ValueError("ttt_loss_init required for signal='ttt_loss_init'")
            signal_value = ttt_loss_init
        elif self.signal == "ttt_improvement":
            if ttt_loss_init is None or ttt_loss_final is None:
                raise ValueError(
                    "ttt_loss_init and ttt_loss_final required for signal='ttt_improvement'"
                )
            signal_value = ttt_loss_init - ttt_loss_final
        else:
            raise ValueError(f"Unknown signal: {self.signal}")

        should_update = jnp.asarray(signal_value) > self.threshold

        return GatingDecision(
            should_update=should_update,
            gating_scale=should_update.astype(jnp.float32),
            metrics={
                "signal": self.signal,
                "signal_value": float(signal_value),
                "threshold": self.threshold,
            },
        )
