"""
Policy trainer using PID-Lagrangian PPO.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..models.policy import compute_gae
from .pid_lagrangian import PIDLagrangianPPO, ppo_update_step
from .ttt_trainer import TrainState


@dataclass
class PolicyTrainer:
    """
    Trainer for adaptive TTT policy.

    Attributes:
        ppo: PID-Lagrangian PPO algorithm
        rollout_length: Length of rollout for data collection
    """
    ppo: PIDLagrangianPPO
    rollout_length: int = 256

    def collect_rollout(
        self,
        policy_state: TrainState,
        ttt_model: Any,
        data_iterator: Iterator,
        rng: jax.random.PRNGKey,
    ) -> dict[str, jnp.ndarray]:
        """
        Collect rollout using current policy.

        Args:
            policy_state: Policy training state
            ttt_model: TTT model for evaluation
            data_iterator: Data iterator
            rng: Random key

        Returns:
            Rollout dictionary with experiences
        """
        features_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        costs_list = []
        values_list = []
        dones_list = []

        budget_used = 0.0

        for _i in range(self.rollout_length):
            # Get next batch
            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            # Extract features (simplified - in practice, use proper feature extraction)
            features = jnp.mean(batch['chunks'], axis=(1, 2))  # [batch, chunk_size] -> [batch]
            features = features[:, None]  # [batch, 1]

            # Get policy action
            rng, action_rng = jax.random.split(rng)
            policy_outputs = policy_state.apply_fn(
                {'params': policy_state.params},
                features,
                rngs={'action': action_rng},
            )

            action = policy_outputs['action']
            log_prob = policy_outputs['log_prob']
            value = policy_outputs['value']

            # Perform TTT with selected action
            # (Simplified - in practice, use actual TTT update)
            reward = jnp.ones_like(action).astype(jnp.float32)  # Placeholder
            cost = jnp.array([1.0, 3.0, 5.0, 12.0])[action]  # Cost per action

            # Store experience
            features_list.append(features)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            costs_list.append(cost)
            values_list.append(value)
            dones_list.append(jnp.zeros_like(action))

            budget_used += jnp.sum(cost)

        # Convert to arrays
        rollout = {
            'features': jnp.concatenate(features_list, axis=0),
            'actions': jnp.concatenate(actions_list, axis=0),
            'old_log_probs': jnp.concatenate(log_probs_list, axis=0),
            'rewards': jnp.concatenate(rewards_list, axis=0),
            'costs': jnp.concatenate(costs_list, axis=0),
            'values': jnp.concatenate(values_list, axis=0),
            'dones': jnp.concatenate(dones_list, axis=0),
        }

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rewards=rollout['rewards'],
            values=rollout['values'],
            dones=rollout['dones'],
        )

        rollout['advantages'] = advantages
        rollout['returns'] = returns

        return rollout

    def train_policy(
        self,
        policy_state: TrainState,
        rollout: dict[str, jnp.ndarray],
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> tuple[TrainState, dict[str, float]]:
        """
        Train policy using collected rollout.

        Args:
            policy_state: Policy training state
            rollout: Collected rollout data
            num_epochs: Number of PPO epochs
            batch_size: Mini-batch size

        Returns:
            updated_state: Updated policy state
            metrics: Training metrics
        """
        num_samples = len(rollout['features'])

        all_metrics = []

        for epoch in range(num_epochs):
            # Shuffle data
            perm = jax.random.permutation(
                jax.random.PRNGKey(epoch),
                num_samples,
            )

            # Mini-batch updates
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i + batch_size]

                batch = {
                    key: value[batch_idx]
                    for key, value in rollout.items()
                }

                # PPO update
                policy_state, metrics = ppo_update_step(
                    state=policy_state,
                    batch=batch,
                    ppo=self.ppo,
                )

                all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            key: jnp.mean(jnp.array([m[key] for m in all_metrics]))
            for key in all_metrics[0].keys()
        }

        # Update PID controller
        avg_cost = jnp.mean(rollout['costs'])
        self.ppo = self.ppo.update_pid(avg_cost)
        avg_metrics['avg_cost'] = avg_cost
        avg_metrics['budget_used'] = jnp.sum(rollout['costs'])

        return policy_state, avg_metrics
