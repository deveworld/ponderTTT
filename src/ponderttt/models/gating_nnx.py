"""
Gating network for differentiable TTT decisions.

Includes:
- GatingNetwork: Original continuous gating (Soft Skip)
- BinaryGatingNetwork: Hard Skip with Gumbel-Softmax for differentiable binary decisions
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class GatingConfig:
    """Configuration for gating network."""

    feature_dim: int = 32
    hidden_dim: int = 64
    dropout_rate: float = 0.0
    scale_output: float = 1.0  # If > 1.0, output is in [0, scale_output]


@dataclass
class BinaryGatingConfig:
    """Configuration for binary gating network (Hard Skip)."""

    feature_dim: int = 32
    hidden_dim: int = 64
    dropout_rate: float = 0.0
    initial_temperature: float = 1.0  # Gumbel-Softmax temperature (annealed during training)
    min_temperature: float = 0.1  # Minimum temperature after annealing
    scale_when_update: float = 1.0  # Gating scale when UPDATE is chosen (default: 1.0)


class GatingNetwork(nnx.Module):
    """
    Gating network for controlling TTT update strength.
    Outputs a scalar s_t in [0, scale_output].
    """

    def __init__(self, config: GatingConfig, rngs: nnx.Rngs):
        """Initialize gating network.

        Args:
            config: Gating configuration
            rngs: Random number generators
        """
        self.config = config

        # Shared feature extraction
        self.input_norm = nnx.LayerNorm(config.feature_dim, rngs=rngs)
        self.fc1 = nnx.Linear(config.feature_dim, config.hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        
        # Output projection (scalar)
        # Initialize bias to negative value to encourage starting with low updates (soft skip)
        self.head = nnx.Linear(
            config.hidden_dim, 
            1, 
            bias_init=nnx.initializers.constant(-2.0),
            rngs=rngs
        )

    def __call__(
        self,
        features: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        """
        Forward pass.

        Args:
            features: Input features [batch, feature_dim]
            train: Whether to enable dropout

        Returns:
            Gating scalar [batch, 1]
        """
        # Flatten inputs
        B = features.shape[0]
        x = features.astype(jnp.float32).reshape(B, -1)

        # Normalize features
        x = self.input_norm(x)

        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        
        x = self.fc2(x)
        x = jax.nn.relu(x)

        logits = self.head(x)
        
        # Sigmoid to [0, 1]
        gate = jax.nn.sigmoid(logits)

        # Scale if needed (e.g. to [0, 4])
        if self.config.scale_output != 1.0:
            gate = gate * self.config.scale_output

        return gate


def gumbel_softmax(
    logits: jax.Array,
    temperature: float,
    hard: bool = False,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    """
    Gumbel-Softmax distribution for differentiable discrete sampling.

    Args:
        logits: Unnormalized log probabilities [batch, num_classes]
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, use straight-through estimator for hard samples
        rng_key: JAX random key (required for sampling)

    Returns:
        Samples from Gumbel-Softmax distribution [batch, num_classes]
    """
    if rng_key is None:
        # Deterministic mode: just use softmax
        return jax.nn.softmax(logits / temperature, axis=-1)

    # Sample Gumbel noise
    gumbel_noise = jax.random.gumbel(rng_key, logits.shape)

    # Add noise and apply softmax
    y_soft = jax.nn.softmax((logits + gumbel_noise) / temperature, axis=-1)

    if hard:
        # Straight-through estimator: hard in forward, soft in backward
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
        # Stop gradient on the difference, so backward uses y_soft
        y = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
        return y
    else:
        return y_soft


class BinaryGatingNetwork(nnx.Module):
    """
    Binary gating network for Hard Skip decisions.

    Uses Gumbel-Softmax for differentiable binary (SKIP/UPDATE) decisions.
    - During training: soft samples with temperature annealing
    - During evaluation: hard argmax decisions

    Output:
    - decision: 0 (SKIP) or 1 (UPDATE)
    - gating_scale: 0.0 (SKIP) or scale_when_update (UPDATE)
    """

    def __init__(self, config: BinaryGatingConfig, rngs: nnx.Rngs):
        """Initialize binary gating network.

        Args:
            config: Binary gating configuration
            rngs: Random number generators
        """
        self.config = config
        self.rngs = rngs

        # Shared feature extraction
        self.input_norm = nnx.LayerNorm(config.feature_dim, rngs=rngs)
        self.fc1 = nnx.Linear(config.feature_dim, config.hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

        # Binary output: 2 logits for SKIP (0) and UPDATE (1)
        # Initialize with NEUTRAL bias - let the gating loss determine the balance
        self.head = nnx.Linear(
            config.hidden_dim,
            2,  # [SKIP, UPDATE]
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        # Neutral initialization: 50/50 probability initially
        self.head.bias = nnx.Param(jnp.array([0.0, 0.0]))

    def __call__(
        self,
        features: jax.Array,
        train: bool = False,
        temperature: float | None = None,
        rng_key: jax.Array | None = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass.

        Args:
            features: Input features [batch, feature_dim]
            train: Whether in training mode
            temperature: Gumbel-Softmax temperature (uses config default if None)
            rng_key: Random key for Gumbel sampling (required for training)

        Returns:
            Tuple of:
            - gating_scale: [batch, 1] - 0.0 for SKIP, scale_when_update for UPDATE
            - decision_probs: [batch, 2] - soft probabilities [P(SKIP), P(UPDATE)]
            - decision_hard: [batch] - hard decisions (0=SKIP, 1=UPDATE)
        """
        B = features.shape[0]
        x = features.astype(jnp.float32).reshape(B, -1)

        # Feature extraction
        x = self.input_norm(x)
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=not train)
        x = self.fc2(x)
        x = jax.nn.relu(x)

        # Get logits for SKIP/UPDATE
        logits = self.head(x)  # [batch, 2]

        # Temperature
        temp = temperature if temperature is not None else self.config.initial_temperature

        if train and rng_key is not None:
            # Training: Gumbel-Softmax with straight-through estimator
            decision_probs = gumbel_softmax(logits, temp, hard=True, rng_key=rng_key)
        else:
            # Evaluation: deterministic softmax (or hard argmax)
            decision_probs = jax.nn.softmax(logits / temp, axis=-1)

        # Hard decision (for logging and actual skip logic)
        decision_hard = jnp.argmax(logits, axis=-1)  # [batch]

        # Gating scale: 0 for SKIP, scale_when_update for UPDATE
        # Use soft probabilities for differentiable training
        update_prob = decision_probs[:, 1:2]  # [batch, 1] - probability of UPDATE
        gating_scale = update_prob * self.config.scale_when_update

        return gating_scale, decision_probs, decision_hard

    def get_decision(self, features: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Get hard decision for evaluation (no sampling).

        Args:
            features: Input features [batch, feature_dim]

        Returns:
            Tuple of:
            - gating_scale: [batch, 1] - 0.0 for SKIP, scale_when_update for UPDATE
            - decision: [batch] - hard decisions (0=SKIP, 1=UPDATE)
        """
        gating_scale, _, decision_hard = self(features, train=False)

        # For evaluation, use hard decision directly
        hard_scale = jnp.where(
            decision_hard[:, None] == 1,
            self.config.scale_when_update,
            0.0
        )

        return hard_scale, decision_hard