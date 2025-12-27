#!/usr/bin/env python3
"""
Analyze Multiple Gating Signal Candidates for PonderTTT.

This script evaluates multiple candidate proxy signals for TTT gating,
using the same methodology as compare_methods.py for consistency.

Signals analyzed:
1. Reconstruction Loss (ttt_loss_init - Full-Sequence)
2. Reconstruction Loss (ttt_loss_step_0 - Last Token)
3. Output Entropy (prediction uncertainty)
4. Token Confidence (mean max probability)
5. TTT Improvement (step_0 - step_1)

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
from typing import Dict, List
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ponderttt.models.base_model_nnx import load_ttt_model, TTTTransformerLM
from ponderttt.data.dataset import create_data_iterator
from ponderttt.data.tokenization import get_tokenizer
from ponderttt.utils import cross_entropy_loss
from flax import nnx


# === JIT-compiled Helpers (same as compare_methods.py) ===


@nnx.jit
def jit_ttt_forward_with_stats(
    model: TTTTransformerLM,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
):
    """Run TTT forward and return loss + ttt_stats.

    Exactly matches compare_methods.py methodology.
    """
    # SKIP path (no TTT)
    out_skip = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    loss_skip = cross_entropy_loss(
        out_skip["logits"][:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )

    # UPDATE path (with TTT)
    out_update = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    loss_update = cross_entropy_loss(
        out_update["logits"][:, :-1],
        input_ids[:, 1:],
        attention_mask[:, 1:],
    )

    # TTT internal stats (take mean across heads/batch)
    ttt_stats = out_update.get("ttt_stats", {})
    # Full-Sequence Reconstruction Loss (better for large models, r=0.77@XL)
    ttt_loss_init = ttt_stats.get("ttt_loss_init", jnp.array(0.0))
    ttt_loss_step_0 = ttt_stats.get("ttt_loss_step_0", jnp.array(0.0))
    ttt_loss_step_1 = ttt_stats.get("ttt_loss_step_1", jnp.array(0.0))

    # Ensure scalars by taking mean (TTT stats are per-head arrays)
    ttt_loss_init = jnp.mean(ttt_loss_init)
    ttt_loss_step_0 = jnp.mean(ttt_loss_step_0)
    ttt_loss_step_1 = jnp.mean(ttt_loss_step_1)

    # Compute entropy and confidence on skip logits
    logits_skip = out_skip["logits"]
    probs = jax.nn.softmax(logits_skip, axis=-1)
    log_probs = jnp.log(probs + 1e-10)
    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
    token_confidence = probs.max(axis=-1).mean()

    return (
        loss_skip,
        loss_update,
        ttt_loss_init,
        ttt_loss_step_0,
        ttt_loss_step_1,
        entropy,
        token_confidence,
    )


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
    print(f"Batch Size: {args.batch_size}")
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
        vocab_size=tokenizer.get_vocab_size(),
    )
    print(f"Model loaded: {args.model_scale} ({model_name})")

    # Create data iterator
    print("\nCreating data iterator...")
    required_examples = args.num_batches * args.batch_size * 2  # 2 chunks per example
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=1024,  # Same as compare_methods.py
        chunk_size=512,  # Same as compare_methods.py
        language=args.language,
        split=args.split,
        max_examples=required_examples,
    )

    # Collect results
    results = []
    total_chunks = 0

    print(f"\nAnalyzing {args.num_batches} batches...")
    target_chunks = args.num_batches * 2  # 2 chunks per batch
    for batch in tqdm(data_iter, total=args.num_batches, desc="Analyzing"):
        if total_chunks >= target_chunks:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        for c_idx in range(num_chunks):
            chunk_ids = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]

            # Use LOCAL position IDs (0 to chunk_len-1) for each chunk
            # This is exactly what compare_methods.py does
            chunk_len = chunk_ids.shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids, chunk_ids.shape)

            # Call JIT function (same as compare_methods.py)
            (
                loss_skip,
                loss_update,
                ttt_loss_init,
                ttt_loss_step_0,
                ttt_loss_step_1,
                entropy,
                token_confidence,
            ) = jit_ttt_forward_with_stats(model, chunk_ids, chunk_mask, position_ids)

            # Convert to Python floats
            loss_skip = float(loss_skip)
            loss_update = float(loss_update)
            ttt_loss_init = float(ttt_loss_init)
            ttt_loss_step_0 = float(ttt_loss_step_0)
            ttt_loss_step_1 = float(ttt_loss_step_1)
            entropy = float(entropy)
            token_confidence = float(token_confidence)

            # Oracle advantage (positive = update helps)
            oracle_advantage = loss_skip - loss_update

            # Valid ratio for is_real_code
            valid_ratio = float(jnp.sum(chunk_mask)) / chunk_len
            is_real_code = valid_ratio > 0.1

            results.append(
                {
                    "oracle_advantage": oracle_advantage,
                    "loss_skip": loss_skip,
                    "loss_update": loss_update,
                    "recon_loss_full_seq": ttt_loss_init,  # Full-Sequence Reconstruction
                    "recon_loss_last_token": ttt_loss_step_0,  # Last Token Reconstruction
                    "ttt_improvement": ttt_loss_step_0 - ttt_loss_step_1,
                    "output_entropy": entropy,
                    "token_confidence": token_confidence,
                    "is_real_code": is_real_code,
                    "valid_ratio": valid_ratio,
                    "chunk_idx": total_chunks,
                }
            )

            total_chunks += 1

    print(f"\nTotal chunks analyzed: {total_chunks}")

    # Filter to real code only
    real_code_results = [r for r in results if r["is_real_code"]]
    print(f"Real code chunks: {len(real_code_results)}")

    # === Correlation Analysis ===
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (Real Code Only)")
    print("=" * 60)

    signals = [
        ("recon_loss_full_seq", "Recon Loss (Full Seq)", "ttt_loss_init"),
        ("recon_loss_last_token", "Recon Loss (Last Token)", "ttt_loss_step_0"),
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
