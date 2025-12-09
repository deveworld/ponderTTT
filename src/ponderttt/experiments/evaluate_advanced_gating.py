#!/usr/bin/env python3
"""
Evaluate advanced multi-signal gating for Run phase.

Compares:
1. Single-signal (TTT improvement only) - Walk phase baseline
2. Multi-signal (TTT + entropy + confidence)
3. Budget-aware gating

Usage:
    python -m ponderttt.experiments.evaluate_advanced_gating \
        --checkpoint outputs/baselines/125m_update1/checkpoints/checkpoint_100000/ \
        --num_batches 1000
"""

import argparse
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ..data import create_data_iterator, get_tokenizer
from ..models import (
    TTTModel,
    load_ttt_model,
    create_advanced_gating,
    AdvancedGatingNetwork,
)
from ..utils.checkpointing import load_checkpoint


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    checkpoint_path: str
    model_size: str = "125m"
    chunk_size: int = 512
    batch_size: int = 1
    num_batches: int = 1000
    skip_examples: int = 160000
    target_update_rate: float = 0.5
    seed: int = 42


def compute_signals(
    model: TTTModel,
    batch: dict,
    gating: AdvancedGatingNetwork,
) -> dict:
    """Compute all gating signals for a batch.

    Returns:
        Dictionary with signals and decisions
    """
    chunks = batch["chunks"]
    mask = batch.get("chunk_attention_mask")

    # Forward pass with TTT to get ttt_improvement
    output = model(chunks, attention_mask=mask, use_ttt=True)

    # Get TTT improvement
    ttt_stats = output.get("ttt_stats", {})
    ttt_loss_0 = ttt_stats.get("ttt_loss_step_0", 0.0)
    ttt_loss_1 = ttt_stats.get("ttt_loss_step_1", 0.0)
    ttt_improvement = ttt_loss_0 - ttt_loss_1

    if isinstance(ttt_improvement, (int, float)):
        ttt_improvement = jnp.array([ttt_improvement])
    elif ttt_improvement.ndim == 0:
        ttt_improvement = ttt_improvement[None]

    # Get logits for entropy/confidence
    logits = output["logits"]

    # Get gating decision
    result = gating(
        ttt_improvement=ttt_improvement,
        logits=logits,
        update_stats=True,
        return_signals=True,
    )

    return {
        "ttt_improvement": ttt_improvement,
        "signals": result["signals"],
        "decision": result["decision"],
        "probability": result["probability"],
    }


def evaluate_gating_method(
    model: TTTModel,
    dataset,
    gating: AdvancedGatingNetwork,
    num_batches: int,
    target_rate: float,
) -> dict:
    """Evaluate a gating method against oracle.

    Returns:
        Dictionary with metrics
    """
    # Collect all samples first
    all_losses_skip = []
    all_losses_update = []
    all_decisions = []
    all_ttt_improvements = []

    logger.info("Collecting samples and computing oracle...")

    for i, batch in enumerate(dataset):
        if i >= num_batches:
            break

        if i % 100 == 0:
            logger.info(f"  Batch {i}/{num_batches}")

        chunks = batch["chunks"]
        mask = batch.get("chunk_attention_mask")

        # Get model outputs and signals
        signals_result = compute_signals(model, batch, gating)
        all_decisions.append(signals_result["decision"][0])
        all_ttt_improvements.append(float(signals_result["ttt_improvement"][0]))

        # Oracle: compute both skip and update losses
        # Skip loss
        skip_output = model(chunks, attention_mask=mask, use_ttt=False)
        skip_logits = skip_output["logits"]
        skip_loss = compute_cross_entropy_loss(skip_logits, chunks)
        all_losses_skip.append(float(skip_loss))

        # Update loss (need fresh model state, but for simplicity use direct update)
        update_output = model(chunks, attention_mask=mask, use_ttt=True)
        update_logits = update_output["logits"]
        update_loss = compute_cross_entropy_loss(update_logits, chunks)
        all_losses_update.append(float(update_loss))

    # Convert to arrays
    all_losses_skip = jnp.array(all_losses_skip)
    all_losses_update = jnp.array(all_losses_update)
    all_decisions = jnp.array(all_decisions)
    all_ttt_improvements = jnp.array(all_ttt_improvements)

    # Oracle advantage
    oracle_advantage = all_losses_skip - all_losses_update

    # Oracle decisions (top-k by advantage)
    k = int(len(oracle_advantage) * target_rate)
    oracle_top_k = jnp.argsort(oracle_advantage)[::-1][:k]
    oracle_decisions = jnp.zeros(len(oracle_advantage), dtype=bool)
    oracle_decisions = oracle_decisions.at[oracle_top_k].set(True)

    # Compute metrics
    # Method loss: weighted by decision
    method_loss = jnp.where(all_decisions, all_losses_update, all_losses_skip).mean()

    # Random skip loss
    random_loss = (all_losses_skip + all_losses_update).mean() / 2

    # Oracle loss
    oracle_loss = jnp.where(oracle_decisions, all_losses_update, all_losses_skip).mean()

    # Decision accuracy (overlap with oracle)
    decision_overlap = (all_decisions == oracle_decisions).mean()

    # Update rate
    actual_rate = all_decisions.mean()

    # Oracle capture rate
    oracle_gap = float(random_loss - oracle_loss)
    method_gap = float(random_loss - method_loss)
    capture_rate = method_gap / oracle_gap if oracle_gap > 0 else 0.0

    # Correlation with oracle
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(all_ttt_improvements, oracle_advantage)

    return {
        "method_loss": float(method_loss),
        "random_loss": float(random_loss),
        "oracle_loss": float(oracle_loss),
        "decision_accuracy": float(decision_overlap),
        "update_rate": float(actual_rate),
        "capture_rate": capture_rate,
        "correlation": correlation,
    }


def compute_cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute cross-entropy loss."""
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    # Flatten
    B, T = shift_targets.shape
    vocab_size = shift_logits.shape[-1]

    flat_logits = shift_logits.reshape(-1, vocab_size)
    flat_targets = shift_targets.reshape(-1)

    # Cross entropy
    log_probs = jax.nn.log_softmax(flat_logits)
    target_log_probs = log_probs[jnp.arange(B * T), flat_targets]

    return -target_log_probs.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="125m")
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--skip_examples", type=int, default=160000)
    parser.add_argument("--target_rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        skip_examples=args.skip_examples,
        target_update_rate=args.target_rate,
        seed=args.seed,
    )

    logger.info(f"Configuration: {config}")

    # Initialize model
    rngs = nnx.Rngs(config.seed)

    logger.info("Loading model...")
    model, _ = load_ttt_model(
        model_name=f"gpt2-{config.model_size}",
        seed=config.seed,
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {config.checkpoint_path}")
    ckpt = load_checkpoint(config.checkpoint_path)
    if ckpt:
        from ..utils.checkpointing import unwrap_state
        model_state = unwrap_state(ckpt["state"]["model"])
        if "fast_layer" in model_state:
            nnx.update(model.fast_layer, model_state["fast_layer"])
        if "fast_norm" in model_state:
            nnx.update(model.fast_norm, model_state["fast_norm"])
        logger.info("Checkpoint loaded successfully")

    # Initialize tokenizer for dataset creation
    logger.info("Initializing tokenizer...")
    tokenizer = get_tokenizer()

    # Methods to compare
    methods = {
        "ttt_only": create_advanced_gating(
            mode="threshold",
            use_entropy=False,
            use_token_confidence=False,
            target_update_rate=config.target_update_rate,
            rngs=rngs,
        ),
        "multi_signal": create_advanced_gating(
            mode="threshold",
            use_entropy=True,
            use_token_confidence=True,
            target_update_rate=config.target_update_rate,
            rngs=rngs,
        ),
        "budget_aware": create_advanced_gating(
            mode="budget_aware",
            use_entropy=True,
            use_token_confidence=True,
            target_update_rate=config.target_update_rate,
            rngs=rngs,
        ),
    }

    # Evaluate each method
    results = {}
    for name, gating in methods.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"{'='*60}")

        # Create fresh dataset iterator for each method
        dataset_iter = create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            language="Python",
            chunk_size=config.chunk_size,
            batch_size=config.batch_size,
            skip_examples=config.skip_examples,
            cache_data=True,
        )

        metrics = evaluate_gating_method(
            model=model,
            dataset=dataset_iter,
            gating=gating,
            num_batches=config.num_batches,
            target_rate=config.target_update_rate,
        )
        results[name] = metrics

        logger.info(f"Results for {name}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("RUN PHASE: Multi-Signal Gating Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'Loss':>10} {'Capture%':>10} {'Accuracy':>10} {'Rate':>10}")
    print("-" * 70)

    for name, metrics in results.items():
        print(f"{name:<20} {metrics['method_loss']:>10.4f} "
              f"{metrics['capture_rate']*100:>10.1f}% "
              f"{metrics['decision_accuracy']*100:>10.1f}% "
              f"{metrics['update_rate']*100:>10.1f}%")

    print("-" * 70)
    print(f"{'Oracle':<20} {results['ttt_only']['oracle_loss']:>10.4f} "
          f"{'100.0%':>10} {'100.0%':>10} {'50.0%':>10}")
    print(f"{'Random':<20} {results['ttt_only']['random_loss']:>10.4f} "
          f"{'0.0%':>10} {'50.0%':>10} {'50.0%':>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
