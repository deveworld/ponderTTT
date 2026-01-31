"""
Random gating: Update with probability p.
"""

import jax
import jax.numpy as jnp

from .base import GatingDecision, GatingStrategy


class RandomGating(GatingStrategy):
    """Random gating with fixed probability.

    Updates with probability p, providing a stochastic baseline.

    Example:
        >>> gating = RandomGating(probability=0.5)
        >>> rng = jax.random.PRNGKey(0)
        >>> decision = gating.decide(rng=rng)
    """

    def __init__(self, probability: float = 0.5):
        """Initialize random gating.

        Args:
            probability: Probability of updating [0.0, 1.0].
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.probability = probability

    def decide(
        self,
        loss_skip: jax.Array | float | None = None,
        ttt_loss_init: jax.Array | float | None = None,
        ttt_loss_final: jax.Array | float | None = None,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> GatingDecision:
        if rng is None:
            raise ValueError("RandomGating requires rng argument")

        should_update = jax.random.bernoulli(rng, p=self.probability)

        return GatingDecision(
            should_update=should_update,
            gating_scale=should_update.astype(jnp.float32),
            metrics={"probability": self.probability},
        )
