"""
Unified Gating Interface for PonderTTT.

Defines the strategy for when to update the TTT layer.
Pivot: Focus on "Loss Skip" (High Loss -> Update) as the primary strategy.
"""

import abc
import dataclasses
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass
class GatingDecision:
    should_update: bool | jax.Array  # Boolean or scalar array (0.0 or 1.0)
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


class GatingStrategy(nnx.Module):
    """Base class for gating strategies."""
    
    @abc.abstractmethod
    def __call__(self, inputs: Any, **kwargs) -> GatingDecision:
        pass


class FixedActionGating(GatingStrategy):
    """Deterministic fixed action (always SKIP or always UPDATE)."""
    
    def __init__(self, action: Literal["SKIP", "UPDATE"], rngs: nnx.Rngs | None = None):
        self.action = action

    def __call__(self, inputs: Any, **kwargs) -> GatingDecision:
        should_update = (self.action == "UPDATE")
        return GatingDecision(should_update=jnp.array(should_update))


class LossSkipGating(GatingStrategy):
    """
    Update if the initial forward pass loss is above a threshold.
    "Surprise -> Adapt"
    """
    
    def __init__(self, threshold: float = 2.0, rngs: nnx.Rngs | None = None):
        # Threshold can be fixed or potentially learnable/dynamic in future
        self.threshold = threshold

    def __call__(self, inputs: Any, loss: jax.Array | float | None = None, **kwargs) -> GatingDecision:
        if loss is None:
            # If loss is not provided (e.g. inference without labels), fallback or error?
            # For now, default to UPDATE if unsure (conservative)
            return GatingDecision(should_update=jnp.array(True), metrics={"reason": "no_loss"})
            
        should_update = (loss > self.threshold)
        return GatingDecision(
            should_update=jnp.asarray(should_update, dtype=jnp.float32),
            metrics={"loss": loss, "threshold": self.threshold}
        )


class RandomGating(GatingStrategy):
    """Randomly update with probability p."""

    def __init__(self, rngs: nnx.Rngs, probability: float = 0.5):
        self.probability = probability

    def __call__(self, inputs: Any, rng: jax.Array | None = None, **kwargs) -> GatingDecision:
        if rng is None:
            return GatingDecision(should_update=jnp.array(True))
        should_update = jax.random.bernoulli(rng, p=self.probability)
        return GatingDecision(should_update=should_update.astype(jnp.float32))
