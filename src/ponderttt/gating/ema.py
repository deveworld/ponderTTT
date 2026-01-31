"""
EMA-based adaptive threshold gating.
"""

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from .base import GatingDecision, GatingStrategy


class EMAGating(GatingStrategy):
    """Adaptive threshold gating with EMA-based adjustment.

    Maintains running statistics of the gating signal and adapts
    the threshold to achieve a target update rate.

    This is the RECOMMENDED strategy for inference as it:
    - Adapts to different data distributions
    - Maintains consistent update rate budget
    - Is fully causal (no lookahead)

    Example:
        >>> gating = EMAGating(target_update_rate=0.5, ema_alpha=0.1)
        >>> for chunk in chunks:
        ...     decision = gating.decide(ttt_loss_init=chunk_loss)
        ...     gating.update_state({"ttt_loss_init": chunk_loss})
    """

    def __init__(
        self,
        target_update_rate: float = 0.5,
        ema_alpha: float = 0.1,
        signal: Literal["loss_skip", "ttt_loss_init"] = "ttt_loss_init",
        initial_threshold: Optional[float] = None,
        calibration_samples: int = 10,
    ):
        """Initialize EMA-based gating.

        Args:
            target_update_rate: Desired fraction of chunks to update [0, 1].
            ema_alpha: Smoothing factor for EMA (higher = more responsive).
            signal: Which signal to use for gating.
            initial_threshold: Starting threshold (None = use first samples).
            calibration_samples: Number of samples for initial calibration.
        """
        self.target_update_rate = target_update_rate
        self.ema_alpha = ema_alpha
        self.signal = signal
        self.initial_threshold = initial_threshold
        self.calibration_samples = calibration_samples

        # Internal state
        self.threshold: float | None = initial_threshold
        self.ema_signal: float | None = None
        self.ema_decision: float | None = None
        self.sample_count: int = 0
        self.calibration_buffer: list[float] = []

    def decide(
        self,
        loss_skip: jax.Array | float | None = None,
        ttt_loss_init: jax.Array | float | None = None,
        ttt_loss_final: jax.Array | float | None = None,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> GatingDecision:
        # Select signal
        if self.signal == "loss_skip":
            if loss_skip is None:
                raise ValueError("loss_skip required")
            signal_value = float(loss_skip)
        else:  # ttt_loss_init
            if ttt_loss_init is None:
                raise ValueError("ttt_loss_init required")
            signal_value = float(ttt_loss_init)

        # Calibration phase: collect samples
        if self.sample_count < self.calibration_samples:
            self.calibration_buffer.append(signal_value)
            self.sample_count += 1

            if self.sample_count == self.calibration_samples:
                # Compute initial threshold at target percentile
                sorted_values = sorted(self.calibration_buffer)
                idx = int((1 - self.target_update_rate) * len(sorted_values))
                self.threshold = sorted_values[min(idx, len(sorted_values) - 1)]
                self.ema_signal = sum(self.calibration_buffer) / len(
                    self.calibration_buffer
                )
                self.ema_decision = self.target_update_rate

            # During calibration, use fixed behavior
            return GatingDecision(
                should_update=jnp.array(True),  # Always update during calibration
                metrics={"phase": "calibration", "sample": self.sample_count},
            )

        # Normal operation: threshold-based decision
        should_update = (
            signal_value > self.threshold if self.threshold is not None else True
        )

        return GatingDecision(
            should_update=jnp.array(should_update),
            gating_scale=jnp.array(1.0 if should_update else 0.0),
            metrics={
                "threshold": self.threshold,
                "signal_value": signal_value,
                "ema_signal": self.ema_signal,
                "ema_decision": self.ema_decision,
            },
        )

    def update_state(self, metrics: dict) -> None:
        """Update EMA and adjust threshold."""
        if self.sample_count < self.calibration_samples:
            return  # Still calibrating

        # Get signal value from metrics
        signal_key = self.signal
        if signal_key not in metrics:
            return

        signal_value = float(metrics[signal_key])
        was_update = float(
            metrics.get(
                "was_update",
                signal_value > self.threshold if self.threshold is not None else True,
            )
        )

        # Update EMAs
        if self.ema_signal is not None:
            self.ema_signal = (
                1 - self.ema_alpha
            ) * self.ema_signal + self.ema_alpha * signal_value
        if self.ema_decision is not None:
            self.ema_decision = (
                1 - self.ema_alpha
            ) * self.ema_decision + self.ema_alpha * was_update

        # Adjust threshold to converge to target rate
        if self.ema_decision is not None and self.threshold is not None:
            rate_error = self.ema_decision - self.target_update_rate
            # If updating too much, raise threshold; if too little, lower it
            adjustment = rate_error * self.ema_alpha * abs(self.threshold)
            self.threshold = max(0.01, self.threshold + adjustment)

    def reset(self) -> None:
        """Reset to initial state."""
        self.threshold = self.initial_threshold
        self.ema_signal = None
        self.ema_decision = None
        self.sample_count = 0
        self.calibration_buffer = []
