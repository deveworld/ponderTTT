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
        use_ttt: bool = ...,
    ) -> dict: ...


def _forward(model: ChunkModel, batch: dict, use_ttt: bool):
    outputs = model(batch["input_ids"], use_ttt=use_ttt)
    logits = outputs["logits"]
    ttt_stats = outputs.get("ttt_stats", {})

    logits_for_loss = logits[:, :-1]
    labels = batch["input_ids"][:, 1:]
    mask = batch["attention_mask"][:, 1:]

    loss = cross_entropy_loss(logits_for_loss, labels, mask)
    return loss, ttt_stats


def metrics_from_loss(loss: jnp.ndarray, ttt_stats: dict | None) -> dict[str, float]:
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


def run_chunk_step(
    model: ChunkModel,
    optimizer: nnx.Optimizer | None,
    batch: dict,
    use_ttt: bool,
    apply_update: bool,
) -> dict[str, float]:
    """
    Run one chunk step (optionally applying gradients).
    """

    def loss_fn(mdl: ChunkModel):
        return _forward(mdl, batch, use_ttt)

    if not apply_update or optimizer is None:
        loss, ttt_stats = loss_fn(model)
        return metrics_from_loss(loss, ttt_stats)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, ttt_stats), grads = grad_fn(model)
    optimizer.update(model, grads)
    return metrics_from_loss(loss, ttt_stats)
