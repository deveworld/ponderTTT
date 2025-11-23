"""
Shared helpers for chunk-level training loops.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from ..utils import cross_entropy_loss


class ChunkModel(Protocol):
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = ...,
        position_ids: jax.Array | None = ...,
        use_ttt: bool = ...,
    ) -> dict: ...


def _forward(model: ChunkModel, batch: dict, use_ttt: bool, ssl_weight: float):
    outputs = model(
        batch["input_ids"], 
        attention_mask=batch.get("attention_mask"),
        position_ids=batch.get("position_ids"),
        use_ttt=use_ttt
    )
    logits = outputs["logits"]
    ttt_stats = outputs.get("ttt_stats", {})

    logits_for_loss = logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:]

    loss = cross_entropy_loss(logits_for_loss, labels, mask)
    if use_ttt and ssl_weight > 0 and ttt_stats:
        ssl_terms = [
            ttt_stats.get("ttt_loss_init"),
            ttt_stats.get("ttt_loss_step_0"),
            ttt_stats.get("ttt_loss_step_1"),
        ]
        # Use .mean() to reduce non-scalar stats and preserve gradients (tracers)
        ssl_values = [x.mean() for x in ssl_terms if x is not None]
        if ssl_values:
            ssl_loss = sum(ssl_values) / len(ssl_values)
            loss = loss + ssl_weight * ssl_loss

    return loss, ttt_stats


def metrics_from_loss(loss: jnp.ndarray, ttt_stats: dict | None) -> dict[str, float]:
    if not jnp.isfinite(loss):
        # In JIT code, raising error might be tricky, usually we return nan loss and handle outside
        # But since metrics_from_loss is called outside JIT usually (after device_get), it's fine.
        # Wait, run_chunk_step calls this. If run_chunk_step is NOT jitted but calls jitted functions,
        # the output loss is a JAX array. We should probably let the caller handle non-finite checks
        # or ensure this function handles JIT tracer delays if inside JIT.
        # For now, assuming run_chunk_step is the python wrapper.
        pass
    
    perplexity = jnp.exp(loss)
    metrics: dict[str, float] = {
        "loss": float(loss),
        "perplexity": float(perplexity),
    }

    if ttt_stats:
        for key, value in ttt_stats.items():
            if hasattr(value, "mean"):
                metrics[f"ttt_{key}"] = float(value.mean())
    return metrics


@nnx.jit
def train_step_jit(
    optimizer: nnx.Optimizer,
    batch: dict,
    use_ttt: bool,
    ssl_weight: float,
) -> tuple[jax.Array, dict]:
    """JIT-compiled training step."""
    def loss_fn(model):
        return _forward(model, batch, use_ttt, ssl_weight)

    (loss, ttt_stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(optimizer.model)
    optimizer.update(optimizer.model, grads)
    return loss, ttt_stats


@nnx.jit
def eval_step_jit(
    model: ChunkModel,
    batch: dict,
    use_ttt: bool,
    ssl_weight: float,
) -> tuple[jax.Array, dict]:
    """JIT-compiled evaluation step."""
    return _forward(model, batch, use_ttt, ssl_weight)


def run_chunk_step(
    model: ChunkModel,
    optimizer: nnx.Optimizer | None,
    batch: dict,
    use_ttt: bool,
    apply_update: bool,
    ssl_weight: float = 0.0,
) -> dict[str, float]:
    """
    Run one chunk step (optionally applying gradients).
    Wraps JIT-compiled steps.
    """
    if apply_update and optimizer is not None:
        loss, ttt_stats = train_step_jit(optimizer, batch, use_ttt, ssl_weight)
    else:
        loss, ttt_stats = eval_step_jit(model, batch, use_ttt, ssl_weight)
    
    # Block until ready to ensure timing is correct if needed, but usually handled by iterator
    # Converting to python float in metrics_from_loss triggers sync.
    return metrics_from_loss(loss, ttt_stats)
