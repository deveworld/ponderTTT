"""
Training utilities for TTT models.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training import train_state


class TrainState(train_state.TrainState):
    """
    Extended train state with additional fields.

    Attributes:
        batch_stats: Batch statistics for normalization (if using BatchNorm)
        dropout_rng: RNG key for dropout
    """
    batch_stats: FrozenDict[str, Any] | None = None
    dropout_rng: jax.Array | None = None


def create_train_state(
    rng: jax.Array, # jax.random.PRNGKey
    model: Any,
    learning_rate: float,
    input_shape: tuple,
) -> TrainState:
    """
    Create initial training state.

    Args:
        rng: Random key
        model: Flax model
        learning_rate: Learning rate
        input_shape: Shape of input for initialization

    Returns:
        Initial training state
    """
    # Split RNG
    params_rng, dropout_rng = jax.random.split(rng)

    # Initialize parameters
    variables = model.init(
        params_rng,
        jnp.ones(input_shape, dtype=jnp.float32),
    )

    params = variables['params']
    batch_stats = variables.get('batch_stats', None)

    # Create optimizer
    tx = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dropout_rng=dropout_rng,
    )


@dataclass
class TTTTrainer:
    """
    Trainer for TTT baseline models.

    Attributes:
        model: Flax model
        learning_rate: Learning rate for TTT updates
    """
    model: Any
    learning_rate: float = 1e-4

    @staticmethod
    @jax.jit
    def _train_step_jit(
        state: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[TrainState, dict[str, float]]:
        """
        JIT-compiled training step for maximum performance.

        Args:
            state: Training state
            batch: Batch of data

        Returns:
            updated_state: Updated training state
            metrics: Training metrics
        """

        def loss_fn(params):
            # Forward pass
            outputs = state.apply_fn(
                {'params': params},
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                deterministic=False,
            )

            # Compute loss (language modeling)
            logits = outputs['logits']
            labels = batch['input_ids'][:, 1:]
            logits = logits[:, :-1]

            # Cross-entropy loss
            vocab_size = logits.shape[-1]
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
            )

            # Mask padding
            if 'attention_mask' in batch:
                mask = batch['attention_mask'][:, 1:]
                loss = loss * mask.reshape(-1)
                loss = jnp.sum(loss) / jnp.sum(mask)
            else:
                loss = jnp.mean(loss)

            return loss, {'loss': loss}

        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Update parameters
        state = state.apply_gradients(grads=grads)

        return state, metrics

    def train_step(
        self,
        state: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[TrainState, dict[str, float]]:
        """
        Perform one training step (delegates to JIT-compiled function).

        Args:
            state: Training state
            batch: Batch of data

        Returns:
            updated_state: Updated training state
            metrics: Training metrics
        """
        return self._train_step_jit(state, batch)

    @staticmethod
    @jax.jit
    def _eval_step_jit(
        state: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, jax.Array]:
        """
        JIT-compiled evaluation step for maximum performance.

        Args:
            state: Training state
            batch: Batch of data

        Returns:
            metrics: Evaluation metrics
        """
        outputs = state.apply_fn(
            {'params': state.params},
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
        )

        logits = outputs['logits']
        labels = batch['input_ids'][:, 1:]
        logits = logits[:, :-1]

        vocab_size = logits.shape[-1]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )

        if 'attention_mask' in batch:
            mask = batch['attention_mask'][:, 1:]
            loss = loss * mask.reshape(-1)
            loss = jnp.sum(loss) / jnp.sum(mask)
        else:
            loss = jnp.mean(loss)

        perplexity = jnp.exp(loss)

        return {
            'loss': loss,
            'perplexity': perplexity,
        }

    def eval_step(
        self,
        state: TrainState,
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, float]:
        """
        Perform one evaluation step (delegates to JIT-compiled function).

        Args:
            state: Training state
            batch: Batch of data

        Returns:
            metrics: Evaluation metrics
        """
        return self._eval_step_jit(state, batch)


def create_learning_rate_schedule(
    base_learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> optax.Schedule:
    """
    Create learning rate schedule with warmup and cosine decay.

    Args:
        base_learning_rate: Peak learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps

    Returns:
        Learning rate schedule function
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=num_warmup_steps,
    )

    decay_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=num_training_steps - num_warmup_steps,
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[num_warmup_steps],
    )

    return schedule_fn
