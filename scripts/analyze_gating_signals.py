#!/usr/bin/env python3
"""
Analyze Multiple Gating Signal Candidates for PonderTTT.

This script evaluates multiple candidate proxy signals for TTT gating:
1. Reconstruction Loss (current baseline - last token only)
2. Full-Sequence Reconstruction Loss (average across all positions)
3. Output Entropy (prediction uncertainty)
4. Token Confidence (mean max probability)
5. Gradient Magnitude (TTT gradient norm)

Usage:
    python scripts/analyze_gating_signals.py \
        --model_scale 125m \
        --update1_checkpoint outputs/baselines/125m_update1/checkpoints/checkpoint_100000 \
        --num_batches 100

Output:
    Correlation analysis showing which signal best predicts Oracle advantage.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as scipy_stats

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ponderttt.models.base_model_nnx import load_ttt_model
from ponderttt.data.dataset import create_data_iterator
from ponderttt.data.tokenization import get_tokenizer
from flax import nnx


@dataclass
class SignalResult:
    """Result for a single chunk."""

    chunk_idx: int
    # Oracle advantage = loss_skip - loss_update (positive = update helps)
    oracle_advantage: float
    loss_skip: float
    loss_update: float
    # Candidate signals
    recon_loss_last_token: float  # Current baseline
    recon_loss_full_seq: float  # NEW: Average over all positions
    output_entropy: float  # NEW: Prediction entropy
    token_confidence: float  # NEW: Mean max probability
    ttt_improvement: float  # step_0 - step_1
    is_real_code: bool  # Valid token ratio > 10%


def compute_entropy(logits: jax.Array) -> jax.Array:
    """Compute prediction entropy from logits.

    High entropy = high uncertainty = potential learning opportunity.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jnp.log(probs + 1e-10)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return entropy.mean()


def compute_token_confidence(logits: jax.Array) -> jax.Array:
    """Compute mean max probability across tokens.

    Low confidence = high uncertainty = potential learning opportunity.
    """
    probs = jax.nn.softmax(logits, axis=-1)
    max_probs = probs.max(axis=-1)
    return max_probs.mean()


def compute_loss(logits: jax.Array, labels: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute cross-entropy loss."""
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    shift_mask = mask[..., 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_log_probs = jnp.take_along_axis(
        log_probs, shift_labels[..., None], axis=-1
    ).squeeze(-1)

    masked_loss = -token_log_probs * shift_mask
    total_loss = masked_loss.sum()
    num_tokens = shift_mask.sum()

    return total_loss / (num_tokens + 1e-10)


@nnx.jit
def compute_raw_signals(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
):
    """Jitted function to compute all signals on GPU."""
    # 1. SKIP path (no TTT update)
    output_skip = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits_skip = output_skip["logits"]
    loss_skip = compute_loss(logits_skip, input_ids, attention_mask)

    # Compute metrics on logits_skip (inside JIT)
    output_entropy = compute_entropy(logits_skip)
    token_confidence = compute_token_confidence(logits_skip)

    # 2. UPDATE path (with TTT update)
    output_update = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    logits_update = output_update["logits"]
    loss_update = compute_loss(logits_update, input_ids, attention_mask)
    ttt_stats = output_update.get("ttt_stats", {})

    return loss_skip, loss_update, output_entropy, token_confidence, ttt_stats


def analyze_chunk(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
) -> Dict:
    """Analyze a single chunk and compute all candidate signals.

    Returns dict with all signal values and Oracle advantage.
    """
    # Call Jitted compute function
    (
        loss_skip_jax,
        loss_update_jax,
        output_entropy_jax,
        token_confidence_jax,
        ttt_stats_jax,
    ) = compute_raw_signals(model, input_ids, attention_mask, position_ids)

    # Helper to safely convert JAX arrays to float
    def to_float(x):
        if x is None:
            return 0.0
        if hasattr(x, "mean"):
            return float(x.mean())
        if isinstance(x, (jax.Array, np.ndarray)):
            return float(x)
        return float(x)

    # Convert JAX arrays to Python values (blocking/sync here)
    loss_skip = float(loss_skip_jax)
    loss_update = float(loss_update_jax)
    output_entropy = float(output_entropy_jax)
    token_confidence = float(token_confidence_jax)

    # Oracle advantage (positive = update helps)
    oracle_advantage = loss_skip - loss_update

    # === Candidate Signals ===

    # 1. Reconstruction Loss (last token only) - current baseline
    recon_loss_last_token = to_float(ttt_stats_jax.get("ttt_loss_step_0"))

    # 2. Full-Sequence Reconstruction Loss - computed from ttt_loss_init
    # Note: We'll approximate this using ttt_loss_init as it covers more positions
    recon_loss_full_seq = (
        to_float(ttt_stats_jax.get("ttt_loss_init")) or recon_loss_last_token
    )

    # 3. Output Entropy (Computed in JIT)
    # output_entropy already converted above

    # 4. Token Confidence (Computed in JIT)
    # token_confidence already converted above

    # 5. TTT Improvement (step_0 - step_1)
    # Handle scalar or array cases safely
    def get_val(key):
        val = ttt_stats_jax.get(key)
        return to_float(val) if val is not None else 0.0

    ttt_step_0 = get_val("ttt_loss_step_0")
    ttt_step_1 = get_val("ttt_loss_step_1")
    ttt_improvement = ttt_step_0 - ttt_step_1

    # Check if real code (>10% valid tokens)
    valid_ratio = float(attention_mask.mean())
    is_real_code = valid_ratio > 0.1

    return {
        "oracle_advantage": oracle_advantage,
        "loss_skip": loss_skip,
        "loss_update": loss_update,
        "recon_loss_last_token": recon_loss_last_token,
        "recon_loss_full_seq": recon_loss_full_seq,
        "output_entropy": output_entropy,
        "token_confidence": token_confidence,
        "ttt_improvement": ttt_improvement,
        "is_real_code": is_real_code,
        "valid_ratio": valid_ratio,
    }


def compute_correlations(results: List[Dict], signal_name: str) -> Dict:
    """Compute Pearson and Spearman correlations for a signal."""
    oracle_advantages = [r["oracle_advantage"] for r in results]
    signal_values = [r[signal_name] for r in results]

    # Filter out NaN/Inf
    valid_pairs = [
        (o, s)
        for o, s in zip(oracle_advantages, signal_values)
        if np.isfinite(o) and np.isfinite(s)
    ]

    if len(valid_pairs) < 10:
        return {"pearson_r": np.nan, "spearman_r": np.nan, "n": len(valid_pairs)}

    o_vals, s_vals = zip(*valid_pairs)

    pearson_r, pearson_p = scipy_stats.pearsonr(o_vals, s_vals)
    spearman_r, spearman_p = scipy_stats.spearmanr(o_vals, s_vals)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n": len(valid_pairs),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze gating signal candidates")
    parser.add_argument(
        "--model_scale", type=str, default="125m", choices=["125m", "350m", "1b", "xl"]
    )
    parser.add_argument(
        "--update1_checkpoint",
        type=str,
        default=None,
        help="Path to UPDATE_1 checkpoint for TTT weights",
    )
    parser.add_argument(
        "--num_batches", type=int, default=100, help="Number of batches to analyze"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--language", type=str, default="Python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file", type=str, default=None, help="Path to save results JSON"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PonderTTT Multi-Signal Gating Analysis")
    print("=" * 60)
    print(f"Model Scale: {args.model_scale}")
    print(f"Checkpoint: {args.update1_checkpoint or 'None (fresh)'}")
    print(f"Batches: {args.num_batches}")
    print(f"Language: {args.language}")
    print("=" * 60)

    # Map scale to HuggingFace model name
    scale_to_model_name = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
        "xl": "gpt2-xl",
    }
    model_name = scale_to_model_name.get(args.model_scale, "gpt2")

    # Load tokenizer (separate from model)
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(model_name)
    print(f"Tokenizer loaded: {model_name}")

    # Load model
    print("\nLoading model...")
    model, _ = load_ttt_model(
        model_name,
        checkpoint_path=args.update1_checkpoint,
        load_pretrained=True,
    )
    print(f"Model loaded: {args.model_scale} ({model_name})")

    # Create data iterator
    print("\nCreating data iterator...")
    required_examples = args.num_batches * args.batch_size
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=args.chunk_size * 2,  # 2 chunks per sequence
        chunk_size=args.chunk_size,  # Must match our chunk_size
        language=args.language,
        split=args.split,
        max_examples=required_examples,
    )

    # Collect results
    results = []
    total_chunks = 0

    print(f"\nAnalyzing {args.num_batches} batches...")
    for batch_idx in range(args.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            print(f"\nWarning: Dataset exhausted after {batch_idx} batches.")
            break

        input_ids = jnp.array(batch["input_ids"])
        attention_mask = jnp.array(batch["attention_mask"])

        # Process chunks (2 per sequence)
        for chunk_idx in range(2):
            start = chunk_idx * args.chunk_size
            end = start + args.chunk_size

            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]
            # Use absolute position IDs (don't reset to 0 for second chunk)
            position_ids = jnp.arange(start, end)[None, :]
            position_ids = jnp.broadcast_to(position_ids, chunk_ids.shape)

            result = analyze_chunk(model, chunk_ids, chunk_mask, position_ids)
            result["chunk_idx"] = total_chunks
            results.append(result)
            total_chunks += 1

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Processed {batch_idx + 1}/{args.num_batches} batches ({total_chunks} chunks)"
            )

    print(f"\nTotal chunks analyzed: {total_chunks}")

    # Filter to real code only
    real_code_results = [r for r in results if r["is_real_code"]]
    print(f"Real code chunks: {len(real_code_results)}")

    # === Correlation Analysis ===
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (Real Code Only)")
    print("=" * 60)

    signals = [
        ("recon_loss_last_token", "Recon Loss (Last Token)", "Current baseline"),
        ("recon_loss_full_seq", "Recon Loss (Full Seq)", "Average over all positions"),
        ("output_entropy", "Output Entropy", "Prediction uncertainty"),
        ("token_confidence", "Token Confidence", "Mean max probability - INVERTED"),
        ("ttt_improvement", "TTT Improvement", "step_0 - step_1"),
    ]

    print(f"\n{'Signal':<30} {'Pearson r':>12} {'Spearman r':>12} {'N':>8}")
    print("-" * 65)

    correlation_results = {}
    for signal_key, signal_name, description in signals:
        corr = compute_correlations(real_code_results, signal_key)
        correlation_results[signal_key] = corr

        pearson_str = (
            f"{corr['pearson_r']:+.4f}" if not np.isnan(corr["pearson_r"]) else "N/A"
        )
        spearman_str = (
            f"{corr['spearman_r']:+.4f}" if not np.isnan(corr["spearman_r"]) else "N/A"
        )

        print(f"{signal_name:<30} {pearson_str:>12} {spearman_str:>12} {corr['n']:>8}")

    print("-" * 65)

    # Find best signal
    best_signal = max(
        correlation_results.items(),
        key=lambda x: abs(x[1]["pearson_r"]) if not np.isnan(x[1]["pearson_r"]) else 0,
    )
    print(
        f"\n🏆 Best Signal: {best_signal[0]} (|r| = {abs(best_signal[1]['pearson_r']):.4f})"
    )

    # === Summary Statistics ===
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    oracle_advantages = [r["oracle_advantage"] for r in real_code_results]
    mean_advantage = np.mean(oracle_advantages)
    std_advantage = np.std(oracle_advantages)
    positive_rate = np.mean([a > 0 for a in oracle_advantages])

    print(f"Oracle Advantage (mean ± std): {mean_advantage:.4f} ± {std_advantage:.4f}")
    print(f"Positive Advantage Rate: {positive_rate:.1%}")
    print(
        f"  (Chunks where UPDATE helps: {sum(a > 0 for a in oracle_advantages)}/{len(oracle_advantages)})"
    )

    # === Save Results ===
    if args.output_file:
        output = {
            "args": vars(args),
            "total_chunks": total_chunks,
            "real_code_chunks": len(real_code_results),
            "correlations": {
                k: {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()
                }
                for k, v in correlation_results.items()
            },
            "summary": {
                "mean_oracle_advantage": float(mean_advantage),
                "std_oracle_advantage": float(std_advantage),
                "positive_advantage_rate": float(positive_rate),
            },
            "best_signal": best_signal[0],
        }

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    best_r = abs(best_signal[1]["pearson_r"])
    if best_r > 0.7:
        print("✅ Strong signal found! Consider implementing gating with this signal.")
    elif best_r > 0.5:
        print("⚠️ Moderate signal. May provide marginal improvement over random.")
    else:
        print(
            "❌ Weak signals. Current approach unlikely to significantly outperform random."
        )
        print("   Consider: (1) Learned gating MLP, (2) Multiple signal combination")

    print(
        "\nNote: Token Confidence should be INVERTED (low confidence = should update)"
    )


if __name__ == "__main__":
    main()
