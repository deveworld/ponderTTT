"""
Gating network for differentiable TTT decisions.
"""

from dataclasses import dataclass
from typing import Optional

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
        self.fc1 = nnx.Linear(config.feature_dim, config.hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        
        # Output projection (scalar)
        self.head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

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