"""
Policy network for adaptive TTT decisions.
"""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass
class PolicyConfig:
    """Configuration for policy network."""
    feature_dim: int = 32
    hidden_dim: int = 128
    num_actions: int = 4  # SKIP, UPDATE_1, UPDATE_2, UPDATE_4
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32


class PolicyNetwork(nn.Module):
    """
    Policy network for deciding TTT actions.

    Uses actor-critic architecture for PPO.

    Attributes:
        config: Policy configuration
    """
    config: PolicyConfig

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
        deterministic: bool = False,
        return_logits: bool = False,
    ) -> dict:
        """
        Forward pass through policy network.

        Args:
            features: Input features [batch, feature_dim]
            deterministic: If True, select argmax action
            return_logits: If True, return raw logits

        Returns:
            Dictionary with:
                - action: Selected actions [batch]
                - log_prob: Log probabilities [batch]
                - value: State values [batch]
                - entropy: Policy entropy [batch]
                - logits: Raw logits [batch, num_actions] (if return_logits=True)
        """
        cfg = self.config

        # Shared feature extraction
        x = nn.Dense(features=cfg.hidden_dim, dtype=cfg.dtype)(features)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        x = nn.Dense(features=cfg.hidden_dim, dtype=cfg.dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)

        # Actor head (policy)
        action_logits = nn.Dense(features=cfg.num_actions, dtype=cfg.dtype, name='actor')(x)

        # Critic head (value)
        value = nn.Dense(features=1, dtype=cfg.dtype, name='critic')(x)
        value = jnp.squeeze(value, axis=-1)

        # Compute action probabilities
        action_probs = jax.nn.softmax(action_logits, axis=-1)

        # Sample or select action
        if deterministic:
            action = jnp.argmax(action_probs, axis=-1)
        else:
            # Sample from categorical distribution
            key = self.make_rng('action')
            action = jax.random.categorical(key, action_logits, axis=-1)

        # Compute log probability of selected action
        log_probs = jax.nn.log_softmax(action_logits, axis=-1)
        selected_log_prob = jnp.take_along_axis(
            log_probs,
            action[:, None],
            axis=-1
        ).squeeze(-1)

        # Compute entropy
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        result = {
            'action': action,
            'log_prob': selected_log_prob,
            'value': value,
            'entropy': entropy,
        }

        if return_logits:
            result['logits'] = action_logits
            result['probs'] = action_probs

        return result

    def evaluate_actions(
        self,
        features: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> dict:
        """
        Evaluate given actions (for PPO update).

        Args:
            features: Input features [batch, feature_dim]
            actions: Actions to evaluate [batch]

        Returns:
            Dictionary with log_prob, value, entropy
        """
        cfg = self.config

        # Forward pass (deterministic=True for evaluation)
        x = nn.Dense(features=cfg.hidden_dim, dtype=cfg.dtype)(features)
        x = nn.relu(x)
        x = nn.Dense(features=cfg.hidden_dim, dtype=cfg.dtype)(x)
        x = nn.relu(x)

        # Get action logits and value
        action_logits = nn.Dense(features=cfg.num_actions, dtype=cfg.dtype, name='actor')(x)
        value = nn.Dense(features=1, dtype=cfg.dtype, name='critic')(x)
        value = jnp.squeeze(value, axis=-1)

        # Compute log probability of given actions
        log_probs = jax.nn.log_softmax(action_logits, axis=-1)
        selected_log_prob = jnp.take_along_axis(
            log_probs,
            actions[:, None],
            axis=-1
        ).squeeze(-1)

        # Compute entropy
        action_probs = jax.nn.softmax(action_logits, axis=-1)
        entropy = -jnp.sum(action_probs * log_probs, axis=-1)

        return {
            'log_prob': selected_log_prob,
            'value': value,
            'entropy': entropy,
        }


def action_to_name(action: int) -> str:
    """Convert action index to name."""
    action_names = ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    return action_names[action]


def action_to_cost(action: int) -> jnp.ndarray:
    """
    Get computational cost for an action.

    Based on PLAN.md:
    - SKIP: 1× (forward only)
    - UPDATE_1: 3× (1 fwd + 2 bwd)
    - UPDATE_2: 5× (2 fwd + 4 bwd)
    - UPDATE_4: 12× (4 fwd + 8 bwd)
    """
    costs = jnp.array([1.0, 3.0, 5.0, 12.0])
    return costs[action]


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
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

    # Compute TD errors (delta)
    next_values = jnp.concatenate([values[1:], jnp.array([0.0])])
    deltas = rewards + gamma * next_values * (1 - dones) - values

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
