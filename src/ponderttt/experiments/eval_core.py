"""
Core evaluation logic for PonderTTT experiments.

This module provides common evaluation primitives that can be used
across different experiment scripts.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, TYPE_CHECKING
import logging

import jax.numpy as jnp
import numpy as np

from .jit_helpers import (
    compute_update_loss,
    compute_both_losses,
    get_ttt_loss_from_stats,
)

if TYPE_CHECKING:
    from ..models import TTTModel
    from ..gating import GatingStrategy


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ChunkResult:
    """Result of evaluating a single chunk."""

    batch_idx: int
    chunk_idx: int
    loss_skip: float
    loss_update: float
    ttt_loss_init: float
    ttt_loss_final: float
    advantage: float  # loss_skip - loss_update (positive = UPDATE better)
    decision: str  # "SKIP" or "UPDATE"

    @property
    def ttt_improvement(self) -> float:
        """Improvement in TTT reconstruction loss."""
        return self.ttt_loss_init - self.ttt_loss_final

    @property
    def oracle_decision(self) -> str:
        """What the Oracle would have chosen."""
        return "UPDATE" if self.advantage > 0 else "SKIP"

    @property
    def is_correct(self) -> bool:
        """Whether decision matches Oracle."""
        return self.decision == self.oracle_decision


@dataclasses.dataclass
class EvalResult:
    """Aggregated evaluation result."""

    chunks: list[ChunkResult]
    method_name: str = ""

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def avg_loss(self) -> float:
        """Average loss according to actual decisions."""
        if not self.chunks:
            return 0.0
        total = 0.0
        for c in self.chunks:
            if c.decision == "UPDATE":
                total += c.loss_update
            else:
                total += c.loss_skip
        return total / len(self.chunks)

    @property
    def avg_loss_skip(self) -> float:
        """Average SKIP path loss (baseline)."""
        if not self.chunks:
            return 0.0
        return sum(c.loss_skip for c in self.chunks) / len(self.chunks)

    @property
    def avg_loss_update(self) -> float:
        """Average UPDATE path loss (full adaptation)."""
        if not self.chunks:
            return 0.0
        return sum(c.loss_update for c in self.chunks) / len(self.chunks)

    @property
    def avg_loss_oracle(self) -> float:
        """Average Oracle loss (best possible)."""
        if not self.chunks:
            return 0.0
        return sum(min(c.loss_skip, c.loss_update) for c in self.chunks) / len(
            self.chunks
        )

    @property
    def oracle_advantage(self) -> float:
        """How much Oracle beats SKIP baseline."""
        return self.avg_loss_skip - self.avg_loss_oracle

    @property
    def update_rate(self) -> float:
        """Fraction of chunks that were updated."""
        if not self.chunks:
            return 0.0
        return sum(1 for c in self.chunks if c.decision == "UPDATE") / len(self.chunks)

    @property
    def oracle_accuracy(self) -> float:
        """Fraction of decisions matching Oracle."""
        if not self.chunks:
            return 0.0
        return sum(1 for c in self.chunks if c.is_correct) / len(self.chunks)

    def correlation(self, signal: str = "ttt_loss_init") -> float:
        """Compute correlation between signal and advantage."""
        if len(self.chunks) < 2:
            return 0.0

        if signal == "ttt_loss_init":
            x = np.array([c.ttt_loss_init for c in self.chunks])
        elif signal == "ttt_improvement":
            x = np.array([c.ttt_improvement for c in self.chunks])
        elif signal == "loss_skip":
            x = np.array([c.loss_skip for c in self.chunks])
        else:
            raise ValueError(f"Unknown signal: {signal}")

        y = np.array([c.advantage for c in self.chunks])

        # Handle constant arrays
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def summary(self) -> dict:
        """Return summary statistics as dict."""
        return {
            "method": self.method_name,
            "total_chunks": self.total_chunks,
            "avg_loss": self.avg_loss,
            "avg_loss_skip": self.avg_loss_skip,
            "avg_loss_update": self.avg_loss_update,
            "avg_loss_oracle": self.avg_loss_oracle,
            "oracle_advantage": self.oracle_advantage,
            "update_rate": self.update_rate,
            "oracle_accuracy": self.oracle_accuracy,
            "corr_ttt_loss": self.correlation("ttt_loss_init"),
            "corr_loss_skip": self.correlation("loss_skip"),
        }


def evaluate_oracle(
    model: "TTTModel",
    data_iter: Iterator,
    num_batches: int,
    chunk_size: int = 512,
    use_local_positions: bool = True,
) -> EvalResult:
    """Evaluate Oracle baseline (upper bound).

    For each chunk, compute both SKIP and UPDATE paths and
    select the one with lower loss (greedy Oracle).

    Args:
        model: TTT model.
        data_iter: Iterator yielding dict with 'chunks' and 'chunk_attention_mask',
                   or tuple (input_ids, attention_mask).
        num_batches: Number of batches to process.
        chunk_size: Tokens per chunk.
        use_local_positions: If True, use local position IDs (0 to chunk_len-1)
                             for each chunk (matches original compare_methods.py).

    Returns:
        EvalResult with all chunk results.
    """
    chunks_results = []
    batch_idx = 0

    for batch_data in data_iter:
        if batch_idx >= num_batches:
            break

        # Handle both dict-style and tuple-style batch data
        if isinstance(batch_data, dict):
            # Dict style: from create_data_iterator
            input_chunks = batch_data["chunks"]  # [batch, num_chunks, chunk_len]
            chunk_masks = batch_data["chunk_attention_mask"]
            num_chunks = input_chunks.shape[1]
        else:
            # Tuple style: (input_ids, attention_mask)
            input_ids, attention_mask = batch_data[:2]
            seq_len = input_ids.shape[1]
            num_chunks = seq_len // chunk_size
            # Will slice below
            input_chunks = None
            chunk_masks = None

        for c_idx in range(num_chunks):
            if input_chunks is not None:
                # Dict style
                chunk_input = input_chunks[:, c_idx]
                chunk_mask = chunk_masks[:, c_idx]
            else:
                # Tuple style
                start = c_idx * chunk_size
                end = start + chunk_size
                chunk_input = input_ids[:, start:end]
                chunk_mask = attention_mask[:, start:end]

            chunk_len = chunk_input.shape[-1]

            # Position IDs: use local (0 to chunk_len-1) like original
            if use_local_positions:
                position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            else:
                # Global position IDs
                pos_start = c_idx * chunk_size
                position_ids = jnp.arange(
                    pos_start, pos_start + chunk_len, dtype=jnp.int32
                )
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            # Compute both paths
            loss_skip, loss_update, ttt_stats = compute_both_losses(
                model, chunk_input, chunk_mask, position_ids
            )

            ttt_loss_init, ttt_loss_final = get_ttt_loss_from_stats(ttt_stats)

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            advantage = loss_skip_val - loss_update_val

            # Oracle chooses the path with lower loss (greedy)
            decision = "UPDATE" if advantage > 0 else "SKIP"

            chunks_results.append(
                ChunkResult(
                    batch_idx=batch_idx,
                    chunk_idx=c_idx,
                    loss_skip=loss_skip_val,
                    loss_update=loss_update_val,
                    ttt_loss_init=ttt_loss_init,
                    ttt_loss_final=ttt_loss_final,
                    advantage=advantage,
                    decision=decision,
                )
            )

        batch_idx += 1

    return EvalResult(chunks=chunks_results, method_name="Oracle")


def evaluate_with_gating(
    model: "TTTModel",
    data_iter: Iterator,
    gating: "GatingStrategy",
    num_batches: int,
    chunk_size: int = 512,
    use_local_positions: bool = True,
    compute_oracle: bool = True,
) -> EvalResult:
    """Evaluate with a gating strategy.

    Args:
        model: TTT model.
        data_iter: Iterator yielding dict with 'chunks' and 'chunk_attention_mask',
                   or tuple (input_ids, attention_mask).
        gating: Gating strategy to use.
        num_batches: Number of batches to process.
        chunk_size: Tokens per chunk.
        use_local_positions: If True, use local position IDs (0 to chunk_len-1).
        compute_oracle: Whether to compute Oracle stats (slower).

    Returns:
        EvalResult with all chunk results.
    """
    chunks_results = []
    batch_idx = 0

    for batch_data in data_iter:
        if batch_idx >= num_batches:
            break

        # Handle both dict-style and tuple-style batch data
        if isinstance(batch_data, dict):
            input_chunks = batch_data["chunks"]
            chunk_masks = batch_data["chunk_attention_mask"]
            num_chunks = input_chunks.shape[1]
        else:
            input_ids, attention_mask = batch_data[:2]
            seq_len = input_ids.shape[1]
            num_chunks = seq_len // chunk_size
            input_chunks = None
            chunk_masks = None

        for c_idx in range(num_chunks):
            if input_chunks is not None:
                chunk_input = input_chunks[:, c_idx]
                chunk_mask = chunk_masks[:, c_idx]
            else:
                start = c_idx * chunk_size
                end = start + chunk_size
                chunk_input = input_ids[:, start:end]
                chunk_mask = attention_mask[:, start:end]

            chunk_len = chunk_input.shape[-1]

            # Position IDs: use local (0 to chunk_len-1) like original
            if use_local_positions:
                position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            else:
                pos_start = c_idx * chunk_size
                position_ids = jnp.arange(
                    pos_start, pos_start + chunk_len, dtype=jnp.int32
                )
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            if compute_oracle:
                # Compute both paths for Oracle comparison
                loss_skip, loss_update, ttt_stats = compute_both_losses(
                    model, chunk_input, chunk_mask, position_ids
                )
                loss_skip_val = float(loss_skip)
                loss_update_val = float(loss_update)
            else:
                # Only compute UPDATE path (stats needed for gating)
                loss_update, ttt_stats = compute_update_loss(
                    model, chunk_input, chunk_mask, position_ids
                )
                loss_skip_val = 0.0  # Unknown
                loss_update_val = float(loss_update)

            ttt_loss_init, ttt_loss_final = get_ttt_loss_from_stats(ttt_stats)

            # Make gating decision
            decision_result = gating.decide(
                loss_skip=loss_skip_val if compute_oracle else None,
                ttt_loss_init=ttt_loss_init,
                ttt_loss_final=ttt_loss_final,
            )

            decision = "UPDATE" if decision_result.should_update else "SKIP"
            advantage = loss_skip_val - loss_update_val

            # Update gating state
            gating.update_state(
                {
                    "ttt_loss_init": ttt_loss_init,
                    "was_update": decision == "UPDATE",
                }
            )

            chunks_results.append(
                ChunkResult(
                    batch_idx=batch_idx,
                    chunk_idx=c_idx,
                    loss_skip=loss_skip_val,
                    loss_update=loss_update_val,
                    ttt_loss_init=ttt_loss_init,
                    ttt_loss_final=ttt_loss_final,
                    advantage=advantage,
                    decision=decision,
                )
            )

        batch_idx += 1

    return EvalResult(chunks=chunks_results, method_name=type(gating).__name__)


def print_eval_summary(result: EvalResult) -> None:
    """Print evaluation summary to console."""
    summary = result.summary()

    print(f"\n{'=' * 50}")
    print(f"Method: {summary['method']}")
    print(f"{'=' * 50}")
    print(f"  Total chunks:      {summary['total_chunks']}")
    print(f"  Avg Loss (method): {summary['avg_loss']:.4f}")
    print(f"  Avg Loss (SKIP):   {summary['avg_loss_skip']:.4f}")
    print(f"  Avg Loss (UPDATE): {summary['avg_loss_update']:.4f}")
    print(f"  Avg Loss (Oracle): {summary['avg_loss_oracle']:.4f}")
    print(f"  Oracle Advantage:  {summary['oracle_advantage']:.4f}")
    print(f"  Update Rate:       {summary['update_rate']:.2%}")
    print(f"  Oracle Accuracy:   {summary['oracle_accuracy']:.2%}")
    print(f"  Corr (TTT Loss):   {summary['corr_ttt_loss']:.4f}")
    print(f"  Corr (Loss Skip):  {summary['corr_loss_skip']:.4f}")
