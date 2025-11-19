"""
Policy network for adaptive TTT decisions in NNX.

Migrated from Linen to NNX for compatibility with NNX-based models.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class PolicyConfig:
    """Configuration for policy network."""

    feature_dim: int = 32
    hidden_dim: int = 128
    num_actions: int = 4  # SKIP, UPDATE_1, UPDATE_2, UPDATE_4
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32


class PolicyNetwork(nnx.Module):
    """
    Policy network for deciding TTT actions.

    Uses actor-critic architecture for PPO.
    """

    def __init__(self, config: PolicyConfig, rngs: nnx.Rngs):
        """Initialize policy network.

        Args:
            config: Policy configuration
            rngs: Random number generators
        """
        self.config = config

        # Shared feature extraction
        self.fc1 = nnx.Linear(config.feature_dim, config.hidden_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(config.dropout_rate, rngs=rngs)

        self.fc2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.dropout2 = nnx.Dropout(config.dropout_rate, rngs=rngs)

        # Actor head (policy)
        self.actor_head = nnx.Linear(config.hidden_dim, config.num_actions, rngs=rngs)

        # Critic head (value)
        self.critic_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

    def forward_shared(self, features: jax.Array, train: bool = False) -> jax.Array:
        """Shared feature extraction.

        Args:
            features: Input features [batch, feature_dim]

        Returns:
            Shared features [batch, hidden_dim]
        """
        x = self.fc1(features)
        x = nnx.relu(x)
        x = self.dropout1(x, deterministic=not train)

        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.dropout2(x, deterministic=not train)

        return x

    def __call__(
        self,
        features: jax.Array,
        deterministic: bool = False,
        return_logits: bool = False,
        rng: Optional[jax.Array] = None,
    ) -> dict:
        """
        Forward pass through policy network.

        Args:
            features: Input features [batch, feature_dim]
            deterministic: If True, select argmax action (not dropout control)
            return_logits: If True, return raw logits
            rng: Random key for action sampling (required if not deterministic)

        Returns:
            Dictionary with:
                - action: Selected actions [batch]
                - log_prob: Log probabilities [batch]
                - value: State values [batch]
                - entropy: Policy entropy [batch]
                - logits: Raw logits [batch, num_actions] (if return_logits=True)
                - probs: Action probabilities [batch, num_actions] (if return_logits=True)

        Note:
            Use model.train() / model.eval() to control dropout, not this parameter.
            The 'deterministic' parameter here only controls action selection.
        """
        # Shared feature extraction
        x = self.forward_shared(features, train=not deterministic)

        # Actor head (policy)
        action_logits = self.actor_head(x)

        # Critic head (value)
        value = self.critic_head(x)
        value = jnp.squeeze(value, axis=-1)

        # Compute action probabilities
        action_probs = jax.nn.softmax(action_logits, axis=-1)

        # Sample or select action
        if deterministic:
            action = jnp.argmax(action_probs, axis=-1)
        else:
            # Sample from categorical distribution
            if rng is None:
                raise ValueError("rng must be provided when deterministic=False")
            action = jax.random.categorical(rng, action_logits, axis=-1)

        # Compute log probability of selected action
        log_probs = jax.nn.log_softmax(action_logits, axis=-1)
        selected_log_prob = jnp.take_along_axis(
            log_probs, action[:, None], axis=-1
        ).squeeze(-1)

        # Compute entropy
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        result = {
            "action": action,
            "log_prob": selected_log_prob,
            "value": value,
            "entropy": entropy,
        }

        if return_logits:
            result["logits"] = action_logits
            result["probs"] = action_probs

        return result

    def evaluate_actions(
        self,
        features: jax.Array,
        actions: jax.Array,
    ) -> dict:
        """
        Evaluate given actions (for PPO update).

        Args:
            features: Input features [batch, feature_dim]
            actions: Actions to evaluate [batch]

        Returns:
            Dictionary with log_prob, value, entropy
        """
        # Forward pass
        x = self.forward_shared(features, train=True)

        # Get action logits and value
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        value = jnp.squeeze(value, axis=-1)

        # Compute log probability of given actions
        log_probs = jax.nn.log_softmax(action_logits, axis=-1)
        selected_log_prob = jnp.take_along_axis(
            log_probs, actions[:, None], axis=-1
        ).squeeze(-1)

        # Compute entropy
        action_probs = jax.nn.softmax(action_logits, axis=-1)
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        return {
            "log_prob": selected_log_prob,
            "value": value,
            "entropy": entropy,
        }


def action_to_name(action: int) -> str:
    """Convert action index to name."""
    action_names = ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    return action_names[action]


def action_to_cost(action: int) -> jnp.ndarray:
    """
    Get computational cost for an action.

    Based on PLAN.md:
    - SKIP: 1x (forward only)
    - UPDATE_1: 3x (1 fwd + 2 bwd)
    - UPDATE_2: 6x (2 fwd + 4 bwd)
    - UPDATE_4: 12x (4 fwd + 8 bwd)
    """
    costs = jnp.array([1.0, 3.0, 6.0, 12.0])
    return costs[action]


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation using jax.lax.scan.

    Args:
        rewards: Rewards [num_steps]
        values: Value estimates [num_steps]
        dones: Done flags [num_steps]
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: Advantage estimates [num_steps]
        returns: Return estimates [num_steps]
    """
    num_steps = len(rewards)
    bootstrap_value = 0.0 if last_value is None else last_value

    extended_values = jnp.concatenate([values, jnp.array([bootstrap_value])])
    deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]

    # Compute GAE using scan (backward pass)
    def scan_fn(gae, t_idx):
        # Process in reverse order
        t = num_steps - 1 - t_idx
        delta = deltas[t]
        done = dones[t]

        # GAE recurrence: gae = delta + gamma * lambda * (1 - done) * gae
        new_gae = delta + gamma * gae_lambda * (1 - done) * gae

        return new_gae, new_gae

    # Run scan in reverse (from last timestep to first)
    _, advantages_reversed = jax.lax.scan(
        scan_fn,
        0.0,  # Initial GAE
        jnp.arange(num_steps),
    )

    # Reverse to get correct order
    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values

    return advantages, returns
