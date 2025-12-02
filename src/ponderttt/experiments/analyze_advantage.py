"""
Analyze advantage distribution and compare Oracle Top-k vs Random selection.

This script answers key questions:
1. What does the advantage distribution look like?
2. How much of total advantage is captured by top k%?
3. What is the theoretical upper bound (Oracle Top-k)?
4. How does Random selection compare?

Usage:
    python -m ponderttt.experiments.analyze_advantage --model_scale 125m --num_samples 1000
"""

import argparse
import json
import math
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import GPT2Model, load_ttt_model
from ..utils import cross_entropy_loss, per_sample_cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze advantage distribution")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m"])
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of chunks to analyze")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    return {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]


def main():
    args = parse_args()

    print("=" * 60)
    print("Advantage Distribution Analysis")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)

    # Load TTT Model
    print("Loading TTT model...")
    ttt_model, ttt_config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # Data Iterator
    chunk_size = args.chunk_size
    seq_length = 1024
    chunks_per_sequence = max(1, seq_length // chunk_size)
    max_examples = math.ceil(args.num_samples / chunks_per_sequence) * args.batch_size

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=max_examples,
        num_workers=args.num_workers,
    )

    # JIT compiled evaluation function
    @jax.jit
    def compute_advantages(
        base_model: GPT2Model,
        fast_layer,
        fast_norm,
        batch: dict,
        tie_word_embeddings: bool = True,
    ):
        """Compute per-sample advantages for a batch."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]
        labels = input_ids

        # Base model forward (SKIP path)
        hidden_states = base_model(input_ids, position_ids=position_ids, train=False)

        # Compute SKIP logits
        if tie_word_embeddings:
            embedding_kernel = jnp.asarray(base_model.wte.embedding)
            logits_skip = hidden_states @ embedding_kernel.T
        else:
            logits_skip = hidden_states

        # TTT forward (UPDATE path)
        hidden_states_normed = fast_norm(hidden_states)
        fast_output, _ = fast_layer(
            hidden_states_normed,
            mask=attention_mask,
            position_ids=position_ids,
            train=False,
            gating_scale=jnp.ones((input_ids.shape[0], 1)),
        )
        adapted_hidden = hidden_states + fast_output

        if tie_word_embeddings:
            logits_update = adapted_hidden @ embedding_kernel.T
        else:
            logits_update = adapted_hidden

        # Per-sample losses
        ce_skip = per_sample_cross_entropy_loss(
            logits_skip[:, :-1], labels[:, 1:], attention_mask[:, 1:]
        )
        ce_update = per_sample_cross_entropy_loss(
            logits_update[:, :-1], labels[:, 1:], attention_mask[:, 1:]
        )

        advantage = ce_skip - ce_update
        return ce_skip, ce_update, advantage

    # Collect advantages
    print(f"Collecting advantages from {args.num_samples} chunks...")
    all_advantages = []
    all_ce_skip = []
    all_ce_update = []

    chunks_collected = 0
    pbar = tqdm(total=args.num_samples)

    for sequence_batch in data_iter:
        if chunks_collected >= args.num_samples:
            break

        chunks = sequence_batch["chunks"]
        masks = sequence_batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        for c_idx in range(num_chunks):
            if chunks_collected >= args.num_samples:
                break

            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
                "position_ids": jnp.arange(
                    c_idx * chunk_size,
                    (c_idx + 1) * chunk_size,
                    dtype=jnp.int32
                )[None, :].repeat(chunks.shape[0], axis=0)
            }

            # Skip mostly-padding chunks
            valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
            if valid_tokens < 16:
                continue

            ce_skip, ce_update, advantage = compute_advantages(
                cast(GPT2Model, ttt_model.base_model),
                ttt_model.fast_layer,
                ttt_model.fast_norm,
                chunk_batch,
                ttt_model.tie_word_embeddings,
            )

            all_advantages.extend(np.array(advantage).tolist())
            all_ce_skip.extend(np.array(ce_skip).tolist())
            all_ce_update.extend(np.array(ce_update).tolist())

            chunks_collected += len(advantage)
            pbar.update(len(advantage))

    pbar.close()

    # Convert to numpy arrays
    advantages = np.array(all_advantages)
    ce_skip = np.array(all_ce_skip)
    ce_update = np.array(all_ce_update)

    print(f"\nCollected {len(advantages)} samples")

    # === ANALYSIS ===
    print("\n" + "=" * 60)
    print("ADVANTAGE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"\nAdvantage Statistics:")
    print(f"  Mean: {advantages.mean():.4f}")
    print(f"  Std:  {advantages.std():.4f}")
    print(f"  Min:  {advantages.min():.4f}")
    print(f"  Max:  {advantages.max():.4f}")
    print(f"  Median: {np.median(advantages):.4f}")

    # What fraction has positive advantage?
    positive_frac = (advantages > 0).mean()
    print(f"\n  Fraction with advantage > 0: {positive_frac:.2%}")

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(advantages, p)
        print(f"    {p}th: {val:.4f}")

    # === TOP-K ANALYSIS ===
    print("\n" + "=" * 60)
    print("TOP-K CAPTURE ANALYSIS")
    print("=" * 60)
    print("How much of total advantage is captured by selecting top k%?\n")

    total_advantage = advantages.sum()
    k_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    topk_results = []
    for k in k_values:
        n_select = max(1, int(len(advantages) * k))
        top_k_indices = np.argsort(advantages)[-n_select:]
        top_k_advantage = advantages[top_k_indices].sum()
        capture_rate = top_k_advantage / total_advantage if total_advantage > 0 else 0
        topk_results.append({
            "k": k,
            "n_selected": n_select,
            "captured_advantage": top_k_advantage,
            "capture_rate": capture_rate,
        })
        print(f"  Top {k*100:5.1f}%: captures {capture_rate*100:6.2f}% of total advantage")

    # === ORACLE vs RANDOM COMPARISON ===
    print("\n" + "=" * 60)
    print("ORACLE TOP-K vs RANDOM SELECTION")
    print("=" * 60)
    print("Expected perplexity improvement under different selection strategies\n")

    # Baseline: No TTT (all SKIP)
    baseline_ce = ce_skip.mean()
    baseline_ppl = math.exp(min(baseline_ce, 10))
    print(f"Baseline (100% SKIP):  CE={baseline_ce:.4f}, PPL={baseline_ppl:.2f}")

    # Full TTT (all UPDATE)
    full_ttt_ce = ce_update.mean()
    full_ttt_ppl = math.exp(min(full_ttt_ce, 10))
    print(f"Full TTT (100% UPDATE): CE={full_ttt_ce:.4f}, PPL={full_ttt_ppl:.2f}")
    print()

    comparison_results = []
    for k in [0.2, 0.3, 0.4, 0.5]:
        n_select = max(1, int(len(advantages) * k))

        # Oracle: select samples with highest advantage
        oracle_indices = np.argsort(advantages)[-n_select:]
        oracle_mask = np.zeros(len(advantages), dtype=bool)
        oracle_mask[oracle_indices] = True

        # Oracle CE: UPDATE for top-k, SKIP for rest
        oracle_ce = np.where(oracle_mask, ce_update, ce_skip).mean()
        oracle_ppl = math.exp(min(oracle_ce, 10))

        # Random: randomly select k%
        np.random.seed(args.seed)
        random_indices = np.random.choice(len(advantages), n_select, replace=False)
        random_mask = np.zeros(len(advantages), dtype=bool)
        random_mask[random_indices] = True

        random_ce = np.where(random_mask, ce_update, ce_skip).mean()
        random_ppl = math.exp(min(random_ce, 10))

        # Theoretical cost
        cost = 1.0 + 2.0 * k  # SKIP=1, UPDATE=3

        improvement_oracle = (baseline_ppl - oracle_ppl) / baseline_ppl * 100
        improvement_random = (baseline_ppl - random_ppl) / baseline_ppl * 100
        oracle_vs_random = (random_ppl - oracle_ppl) / random_ppl * 100

        print(f"Update Rate {k*100:.0f}% (Cost {cost:.1f}x):")
        print(f"  Oracle Top-{k*100:.0f}%:  CE={oracle_ce:.4f}, PPL={oracle_ppl:.2f} ({improvement_oracle:+.1f}% vs baseline)")
        print(f"  Random {k*100:.0f}%:      CE={random_ce:.4f}, PPL={random_ppl:.2f} ({improvement_random:+.1f}% vs baseline)")
        print(f"  Oracle advantage:  {oracle_vs_random:.1f}% better than random")
        print()

        comparison_results.append({
            "k": k,
            "cost": cost,
            "oracle_ce": oracle_ce,
            "oracle_ppl": oracle_ppl,
            "random_ce": random_ce,
            "random_ppl": random_ppl,
            "oracle_vs_random_pct": oracle_vs_random,
        })

    # === SAVE RESULTS ===
    results = {
        "model_scale": args.model_scale,
        "num_samples": len(advantages),
        "advantage_stats": {
            "mean": float(advantages.mean()),
            "std": float(advantages.std()),
            "min": float(advantages.min()),
            "max": float(advantages.max()),
            "median": float(np.median(advantages)),
            "positive_fraction": float(positive_frac),
        },
        "percentiles": {str(p): float(np.percentile(advantages, p)) for p in percentiles},
        "topk_capture": topk_results,
        "oracle_vs_random": comparison_results,
        "baseline_ppl": baseline_ppl,
        "full_ttt_ppl": full_ttt_ppl,
    }

    output_file = output_dir / f"advantage_analysis_{args.model_scale}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Save raw advantages for histogram plotting
    np.save(output_dir / f"advantages_{args.model_scale}.npy", advantages)
    print(f"Raw advantages saved to {output_dir}/advantages_{args.model_scale}.npy")

    # === CONCLUSIONS ===
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    best_k = max(comparison_results, key=lambda x: x["oracle_vs_random_pct"])
    print(f"\n1. Oracle advantage over random is largest at k={best_k['k']*100:.0f}%:")
    print(f"   Oracle is {best_k['oracle_vs_random_pct']:.1f}% better than random")

    print(f"\n2. Advantage distribution:")
    if positive_frac > 0.9:
        print(f"   {positive_frac*100:.0f}% of chunks benefit from TTT (advantage > 0)")
        print("   This explains why naive gating tends to 'always UPDATE'")

    top20_capture = next(r for r in topk_results if r["k"] == 0.2)["capture_rate"]
    print(f"\n3. Top 20% captures {top20_capture*100:.1f}% of total advantage")
    print("   → Selective updating can be very efficient!")

    print("\n4. Recommendation:")
    print(f"   Train with target_update_rate={best_k['k']:.1f} for best discrimination")


if __name__ == "__main__":
    main()
