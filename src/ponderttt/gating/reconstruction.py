"""
TTT Reconstruction Loss-based gating.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from .base import GatingDecision, GatingStrategy


class ReconstructionGating(GatingStrategy):
    """TTT Reconstruction Loss-based gating (PonderTTT method).

    Uses the initial TTT reconstruction loss as the gating signal.
    High reconstruction loss indicates the model's self-supervised
    objective is not well-satisfied, suggesting TTT update will help.

    This is the core method proposed in the PonderTTT paper.

    Modes:
        - "percentile": Update top-k% based on reconstruction loss
        - "threshold": Update if reconstruction loss > threshold
        - "adaptive": Use EMA to adapt threshold (recommended)

    Example:
        >>> gating = ReconstructionGating(mode="adaptive", target_rate=0.5)
        >>> decision = gating.decide(ttt_loss_init=0.8)
    """

    def __init__(
        self,
        mode: str = "adaptive",
        threshold: float = 0.5,
        target_rate: float = 0.5,
        ema_alpha: float = 0.1,
        calibration_batches: int = 2,
    ):
        """Initialize reconstruction gating.

        Args:
            mode: "threshold", "percentile", or "adaptive".
            threshold: Fixed threshold (for mode="threshold").
            target_rate: Target update rate (for mode="adaptive").
            ema_alpha: EMA smoothing factor.
            calibration_batches: Batches for initial calibration.
        """
        self.mode = mode
        self.fixed_threshold = threshold
        self.target_rate = target_rate
        self.ema_alpha = ema_alpha
        self.calibration_batches = calibration_batches

        # Internal state for adaptive mode
        self._threshold: Optional[float] = threshold if mode == "threshold" else None
        self._ema_loss: Optional[float] = None
        self._ema_rate: Optional[float] = None
        self._calibration_losses: list[float] = []
        self._calibrated: bool = mode == "threshold"

    def decide(
        self,
        ttt_loss_init: jax.Array | float | None = None,
        **kwargs,
    ) -> GatingDecision:
        if ttt_loss_init is None:
            raise ValueError("ttt_loss_init required for ReconstructionGating")

        loss_value = float(ttt_loss_init)

        # Calibration phase for adaptive mode
        if self.mode == "adaptive" and not self._calibrated:
            self._calibration_losses.append(loss_value)

            # Check if we have enough samples
            if (
                len(self._calibration_losses) >= self.calibration_batches * 8
            ):  # ~8 chunks per batch
                # Compute threshold at target percentile
                sorted_losses = sorted(self._calibration_losses)
                idx = int((1 - self.target_rate) * len(sorted_losses))
                self._threshold = sorted_losses[min(idx, len(sorted_losses) - 1)]
                self._ema_loss = sum(self._calibration_losses) / len(
                    self._calibration_losses
                )
                self._ema_rate = self.target_rate
                self._calibrated = True

            # During calibration, always update (safe default)
            return GatingDecision(
                should_update=jnp.array(True),
                metrics={
                    "phase": "calibration",
                    "samples": len(self._calibration_losses),
                },
            )

        # Normal operation
        if self._threshold is None:
            self._threshold = self.fixed_threshold

        should_update = loss_value > self._threshold

        return GatingDecision(
            should_update=jnp.array(should_update),
            gating_scale=jnp.array(1.0 if should_update else 0.0),
            metrics={
                "ttt_loss_init": loss_value,
                "threshold": self._threshold,
                "ema_rate": self._ema_rate,
            },
        )

    def update_state(self, metrics: dict) -> None:
        """Update EMA and adapt threshold for adaptive mode."""
        if self.mode != "adaptive" or not self._calibrated:
            return

        loss_value = metrics.get("ttt_loss_init")
        was_update = metrics.get("was_update")

        if loss_value is not None:
            loss_value = float(loss_value)
            # Update EMA of loss
            if self._ema_loss is not None:
                self._ema_loss = (
                    1 - self.ema_alpha
                ) * self._ema_loss + self.ema_alpha * loss_value

        if was_update is not None and self._ema_rate is not None:
            was_update = float(was_update)
            # Update EMA of update rate
            self._ema_rate = (
                1 - self.ema_alpha
            ) * self._ema_rate + self.ema_alpha * was_update

            # Adjust threshold to converge to target rate
            if self._threshold is not None:
                rate_error = self._ema_rate - self.target_rate
                # If updating too much, raise threshold; too little, lower it
                adjustment = (
                    rate_error * self.ema_alpha * max(0.1, abs(self._threshold))
                )
                self._threshold = max(0.01, self._threshold + adjustment)

    def reset(self) -> None:
        """Reset internal state."""
        self._threshold = self.fixed_threshold if self.mode == "threshold" else None
        self._ema_loss = None
        self._ema_rate = None
        self._calibration_losses = []
        self._calibrated = self.mode == "threshold"
