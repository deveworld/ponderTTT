"""
PID-Lagrangian PPO for budget-constrained RL.

Based on: Stooke et al., "Responsive Safety in Reinforcement Learning
by PID Lagrangian Methods", ICML 2020
"""

import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Dict
from dataclasses import dataclass
from flax.core import FrozenDict


@dataclass
class PIDController:
    """
    PID controller for Lagrangian multiplier.

    Attributes:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        lambda_value: Current Lagrangian multiplier
        integral: Integral state
        previous_error: Previous error for derivative
    """
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.01
    lambda_value: float = 1.0
    integral: float = 0.0
    previous_error: float = 0.0

    def update(
        self,
        constraint_violation: float,
        dt: float = 1.0,
    ) -> 'PIDController':
        """
        Update Lagrangian multiplier based on constraint violation.

        Args:
            constraint_violation: Current violation (positive = violated)
            dt: Time step

        Returns:
            Updated PID controller
        """
        error = constraint_violation

        # PID terms
        p_term = self.kp * error
        i_term = self.ki * (self.integral + error * dt)
        d_term = self.kd * (error - self.previous_error) / dt

        # Update lambda
        delta_lambda = p_term + i_term + d_term
        new_lambda = jnp.maximum(0.0, self.lambda_value + delta_lambda)

        return PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            lambda_value=new_lambda,
            integral=self.integral + error * dt,
            previous_error=error,
        )


class PIDLagrangianPPO:
    """
    PPO with PID-controlled Lagrangian constraint enforcement.

    Attributes:
        budget_limit: Maximum allowed cost per episode
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
    """

    def __init__(
        self,
        budget_limit: float,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
    ):
        self.budget_limit = budget_limit
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Initialize PID controller
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)

    def compute_ppo_loss(
        self,
        params: FrozenDict,
        policy_fn: callable,
        features: jnp.ndarray,
        actions: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        costs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute PPO loss with Lagrangian constraint.

        Args:
            params: Policy parameters
            policy_fn: Policy network function
            features: State features [batch, feature_dim]
            actions: Actions taken [batch]
            old_log_probs: Old log probabilities [batch]
            advantages: Advantage estimates [batch]
            returns: Return estimates [batch]
            costs: Costs incurred [batch]

        Returns:
            total_loss: Combined loss
            metrics: Dictionary with loss components
        """
        # Evaluate actions with current policy
        policy_outputs = policy_fn(params, features, actions)

        log_probs = policy_outputs['log_prob']
        values = policy_outputs['value']
        entropy = policy_outputs['entropy']

        # PPO clipped surrogate objective
        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        # Lagrangian adjustment (subtract lambda * cost from advantages)
        lambda_value = self.pid.lambda_value
        lagrangian_advantages = advantages - lambda_value * costs

        # Policy loss
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value loss
        value_loss = jnp.mean((values - returns) ** 2)

        # Entropy bonus
        entropy_loss = -jnp.mean(entropy)

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # Compute approximate KL for monitoring
        approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))

        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': jnp.mean(entropy),
            'approx_kl': approx_kl,
            'lambda': lambda_value,
        }

        return total_loss, metrics

    def update_pid(
        self,
        avg_cost: float,
    ) -> 'PIDLagrangianPPO':
        """
        Update PID controller based on average cost.

        Args:
            avg_cost: Average cost from rollout

        Returns:
            Updated PPO instance
        """
        cost_violation = avg_cost - self.budget_limit
        self.pid = self.pid.update(cost_violation)
        return self


def create_ppo_optimizer(learning_rate: float = 3e-4) -> optax.GradientTransformation:
    """
    Create Adam optimizer for PPO.

    Args:
        learning_rate: Learning rate

    Returns:
        Optax optimizer
    """
    return optax.adam(learning_rate)


def ppo_update_step(
    state: 'TrainState',
    batch: Dict[str, jnp.ndarray],
    ppo: PIDLagrangianPPO,
) -> Tuple['TrainState', Dict[str, jnp.ndarray]]:
    """
    Perform one PPO update step.

    Args:
        state: Training state
        batch: Batch of data
        ppo: PID-Lagrangian PPO instance

    Returns:
        updated_state: Updated training state
        metrics: Training metrics
    """

    def loss_fn(params):
        return ppo.compute_ppo_loss(
            params=params,
            policy_fn=state.apply_fn,
            features=batch['features'],
            actions=batch['actions'],
            old_log_probs=batch['old_log_probs'],
            advantages=batch['advantages'],
            returns=batch['returns'],
            costs=batch['costs'],
        )

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    metrics['loss'] = loss

    return state, metrics
