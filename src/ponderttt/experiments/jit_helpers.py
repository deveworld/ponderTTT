"""
JIT-compiled helper functions for TTT evaluation.

These functions are defined at module level and cached to avoid
recompilation in experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from ..models import TTTTransformerLM


def cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute cross-entropy loss.

    Args:
        logits: [batch, seq, vocab]
        targets: [batch, seq]
        mask: [batch, seq] or None

    Returns:
        Scalar loss value.
    """
    vocab_size = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    losses = -jnp.sum(targets_one_hot * log_probs, axis=-1)

    if mask is not None:
        losses = losses * mask
        return jnp.sum(losses) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        return jnp.mean(losses)


# --- JIT-Compiled Functions ---


@nnx.jit
def compute_skip_loss(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    """Compute loss without TTT update (SKIP path).

    Args:
        model: TTT model.
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        position_ids: [batch, seq_len]

    Returns:
        Scalar loss value.
    """
    out = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits = out["logits"]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    return cross_entropy_loss(shift_logits, shift_labels, shift_mask)


@nnx.jit
def compute_update_loss(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> Tuple[jax.Array, dict]:
    """Compute loss with TTT update (UPDATE path).

    Args:
        model: TTT model.
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        position_ids: [batch, seq_len]

    Returns:
        (loss, ttt_stats) tuple.
    """
    out = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    logits = out["logits"]
    ttt_stats = out.get("ttt_stats", {})

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    loss = cross_entropy_loss(shift_logits, shift_labels, shift_mask)
    return loss, ttt_stats


@nnx.jit
def compute_both_losses(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> Tuple[jax.Array, jax.Array, dict]:
    """Compute both SKIP and UPDATE losses in one call.

    This is more efficient when you need both losses (e.g., Oracle evaluation).

    Args:
        model: TTT model.
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        position_ids: [batch, seq_len]

    Returns:
        (loss_skip, loss_update, ttt_stats) tuple.
    """
    # SKIP path
    out_skip = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits_skip = out_skip["logits"]

    # UPDATE path
    out_update = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    logits_update = out_update["logits"]
    ttt_stats = out_update.get("ttt_stats", {})

    # Shift for next-token prediction
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    loss_skip = cross_entropy_loss(logits_skip[:, :-1, :], shift_labels, shift_mask)
    loss_update = cross_entropy_loss(logits_update[:, :-1, :], shift_labels, shift_mask)

    return loss_skip, loss_update, ttt_stats


@nnx.jit
def compute_gated_loss(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    gating_scale: jax.Array,
) -> Tuple[jax.Array, dict]:
    """Compute loss with explicit gating scale.

    Allows soft gating via gating_scale in [0, 1].

    Args:
        model: TTT model.
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        position_ids: [batch, seq_len]
        gating_scale: Scalar in [0, 1]. 0 = SKIP, 1 = full UPDATE.

    Returns:
        (loss, ttt_stats) tuple.
    """
    out = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
        gating_scale=gating_scale,
    )
    logits = out["logits"]
    ttt_stats = out.get("ttt_stats", {})

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    loss = cross_entropy_loss(shift_logits, shift_labels, shift_mask)
    return loss, ttt_stats


def get_ttt_loss_from_stats(ttt_stats: dict | None) -> Tuple[float, float]:
    """Extract TTT loss values from stats dict.

    Args:
        ttt_stats: Stats dict from model forward pass.

    Returns:
        (ttt_loss_init, ttt_loss_final) tuple.
    """
    if ttt_stats is None:
        return 0.0, 0.0

    # Original compare_methods.py uses "ttt_loss_step_0" as key
    # Also support "ttt_loss_init" as fallback
    ttt_loss_init = ttt_stats.get(
        "ttt_loss_step_0", ttt_stats.get("ttt_loss_init", 0.0)
    )
    ttt_loss_final = ttt_stats.get(
        "ttt_loss_step_1", ttt_stats.get("ttt_loss_final", 0.0)
    )

    # Handle array types - take mean across heads/batch
    if hasattr(ttt_loss_init, "mean"):
        ttt_loss_init = float(ttt_loss_init.mean())
    if hasattr(ttt_loss_final, "mean"):
        ttt_loss_final = float(ttt_loss_final.mean())

    return float(ttt_loss_init), float(ttt_loss_final)
