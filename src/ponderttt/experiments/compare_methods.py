"""
Compare gating methods for adaptive Test-Time Training.

Implemented Methods:
    1. SKIP (Baseline): No TTT updates
    2. UPDATE_1 (Fixed): Always update with 1 gradient step
    3. Random Skip: Randomly select chunks to update
    4. Oracle: Upper bound using ground-truth advantage
    5. PonderTTT (Ours): Reconstruction loss-based gating

Usage:
    python -m ponderttt.experiments.compare_methods --model_scale 125m --budget 2.0
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTModel

from ..gating import (
    FixedGating,
    RandomGating,
    ReconstructionGating,
)
from .jit_helpers import (
    compute_both_losses,
    get_ttt_loss_from_stats,
)

# Global correlation data collector
_correlation_data: dict = {}


# =============================================================================
# Utility Functions
# =============================================================================


def permute_within_chunks(input_ids: jax.Array, seed: int) -> jax.Array:
    """Permute tokens randomly within each chunk."""
    B, L = input_ids.shape
    key = jax.random.PRNGKey(seed)

    def permute_seq(seq, k):
        return jax.random.permutation(k, seq)

    keys = jax.random.split(key, B)
    return jax.vmap(permute_seq)(input_ids, keys)


def get_model_name(model_scale: str) -> str:
    """Convert model scale to HuggingFace model name."""
    return {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
        "xl": "gpt2-xl",
    }[model_scale]


def budget_to_update_rate(budget: float) -> float:
    """Convert budget (cost multiplier) to target update rate."""
    # cost = 1 + 2 * update_rate => update_rate = (budget - 1) / 2
    rate = (budget - 1.0) / 2.0
    return float(min(max(rate, 0.0), 1.0))


# =============================================================================
# Core Evaluation Function
# =============================================================================


def evaluate_method(
    method_name: str,
    model: TTTModel,
    data_iter,
    num_batches: int,
    gating_strategy,
    seed: int = 42,
    shuffle: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a gating method on the given data.

    Returns DataFrame with per-chunk results.
    """
    results = {
        "method": [],
        "loss": [],
        "loss_skip": [],
        "loss_update": [],
        "advantage": [],
        "decision": [],
        "ttt_loss_init": [],
        "cost": [],
    }

    batch_idx = 0
    for batch in tqdm(data_iter, total=num_batches, desc=method_name):
        if batch_idx >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        batch_size = chunks.shape[0]

        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            chunk_len = chunk_input.shape[-1]

            # Apply shuffle if requested
            if shuffle:
                chunk_input = permute_within_chunks(chunk_input, seed + c_idx)

            # Local position IDs
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            # Compute both paths
            loss_skip, loss_update, ttt_stats = compute_both_losses(
                model, chunk_input, chunk_mask, position_ids
            )

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            ttt_loss_init, _ = get_ttt_loss_from_stats(ttt_stats)
            advantage = loss_skip_val - loss_update_val

            # Make gating decision
            decision_result = gating_strategy.decide(
                loss_skip=loss_skip_val,
                ttt_loss_init=ttt_loss_init,
                ttt_loss_final=0.0,
            )
            decision = "UPDATE" if decision_result.should_update else "SKIP"

            # Update gating state
            gating_strategy.update_state(
                {
                    "ttt_loss_init": ttt_loss_init,
                    "was_update": decision == "UPDATE",
                }
            )

            # Record results
            for _ in range(batch_size):
                results["method"].append(method_name)
                results["loss"].append(
                    loss_update_val if decision == "UPDATE" else loss_skip_val
                )
                results["loss_skip"].append(loss_skip_val)
                results["loss_update"].append(loss_update_val)
                results["advantage"].append(advantage)
                results["decision"].append(decision)
                results["ttt_loss_init"].append(ttt_loss_init)
                results["cost"].append(3.0 if decision == "UPDATE" else 1.0)

        batch_idx += 1

    return pd.DataFrame(results)


def evaluate_oracle(
    model: TTTModel,
    data_iter,
    num_batches: int,
    target_update_rate: float = 0.5,
    seed: int = 42,
    shuffle: bool = False,
) -> pd.DataFrame:
    """
    Oracle baseline: select top-k% chunks by advantage.
    """
    # First pass: collect all chunk data
    chunk_data = []

    batch_idx = 0
    for batch in tqdm(data_iter, total=num_batches, desc="Oracle"):
        if batch_idx >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        batch_size = chunks.shape[0]

        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            chunk_len = chunk_input.shape[-1]

            if shuffle:
                chunk_input = permute_within_chunks(chunk_input, seed + c_idx)

            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            loss_skip, loss_update, ttt_stats = compute_both_losses(
                model, chunk_input, chunk_mask, position_ids
            )

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            ttt_loss_init, _ = get_ttt_loss_from_stats(ttt_stats)
            advantage = loss_skip_val - loss_update_val

            chunk_data.append(
                {
                    "batch_idx": batch_idx,
                    "chunk_idx": c_idx,
                    "batch_size": batch_size,
                    "loss_skip": loss_skip_val,
                    "loss_update": loss_update_val,
                    "advantage": advantage,
                    "ttt_loss_init": ttt_loss_init,
                }
            )

        batch_idx += 1

    # Second pass: select top-k by advantage
    num_to_update = max(1, int(len(chunk_data) * target_update_rate))
    sorted_chunks = sorted(chunk_data, key=lambda x: x["advantage"], reverse=True)
    update_set = set(
        (c["batch_idx"], c["chunk_idx"]) for c in sorted_chunks[:num_to_update]
    )

    # Build results
    results = {
        "method": [],
        "loss": [],
        "loss_skip": [],
        "loss_update": [],
        "advantage": [],
        "decision": [],
        "ttt_loss_init": [],
        "cost": [],
    }

    for c in chunk_data:
        key = (c["batch_idx"], c["chunk_idx"])
        decision = "UPDATE" if key in update_set else "SKIP"

        for _ in range(c["batch_size"]):
            results["method"].append("Oracle")
            results["loss"].append(
                c["loss_update"] if decision == "UPDATE" else c["loss_skip"]
            )
            results["loss_skip"].append(c["loss_skip"])
            results["loss_update"].append(c["loss_update"])
            results["advantage"].append(c["advantage"])
            results["decision"].append(decision)
            results["ttt_loss_init"].append(c["ttt_loss_init"])
            results["cost"].append(3.0 if decision == "UPDATE" else 1.0)

    return pd.DataFrame(results)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Compare gating methods")
    parser.add_argument(
        "--model_scale",
        type=str,
        default="125m",
        choices=["125m", "350m", "1b", "xl"],
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="Target budget (cost multiplier)",
    )
    parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=20,
        help="Number of batches",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="Python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--skip_examples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods: skip,update1,random,oracle,ours,all",
    )
    parser.add_argument(
        "--ttt_base_lr",
        type=float,
        default=None,
        help="Override TTT base learning rate",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Method Comparison")
    print("=" * 60)
    print(f"Model: {args.model_scale}")
    print(f"Budget: {args.budget}")
    print(f"Batches: {args.num_eval_batches}")
    print(f"Language: {args.language}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)

    model, _ = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )
    model.eval()

    target_update_rate = budget_to_update_rate(args.budget)
    print(f"Target update rate: {target_update_rate:.1%}")

    # Parse methods to evaluate
    methods_str = args.methods.lower()
    if methods_str == "all":
        methods = ["skip", "update1", "random", "oracle", "ours"]
    else:
        methods = [m.strip() for m in methods_str.split(",")]

    all_results = []

    # Helper to create data iterator
    def make_data_iter():
        return create_data_iterator(
            tokenizer=tokenizer,
            split=args.split,
            batch_size=args.batch_size,
            seq_length=1024,
            chunk_size=512,
            max_examples=args.num_eval_batches * args.batch_size * 2,
            num_workers=args.num_workers,
            language=args.language,
            skip_first_n=args.skip_examples,
        )

    # SKIP baseline
    if "skip" in methods:
        print("\n--- SKIP ---")
        gating = FixedGating(always_update=False)
        df = evaluate_method(
            "SKIP",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        print(f"  Avg loss: {df['loss'].mean():.4f}")

    # UPDATE_1 baseline
    if "update1" in methods:
        print("\n--- UPDATE_1 ---")
        gating = FixedGating(always_update=True)
        df = evaluate_method(
            "UPDATE_1",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        print(f"  Avg loss: {df['loss'].mean():.4f}")

    # Random Skip
    if "random" in methods:
        print("\n--- Random Skip ---")
        gating = RandomGating(update_prob=target_update_rate)
        df = evaluate_method(
            "Random Skip",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        print(f"  Avg loss: {df['loss'].mean():.4f}")
        print(f"  Update rate: {(df['decision'] == 'UPDATE').mean():.1%}")

    # Oracle
    if "oracle" in methods:
        print("\n--- Oracle ---")
        df = evaluate_oracle(
            model,
            make_data_iter(),
            args.num_eval_batches,
            target_update_rate,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        print(f"  Avg loss: {df['loss'].mean():.4f}")
        print(f"  Update rate: {(df['decision'] == 'UPDATE').mean():.1%}")

    # PonderTTT (Ours)
    if "ours" in methods:
        print("\n--- PonderTTT (Ours) ---")
        gating = ReconstructionGating(target_update_rate=target_update_rate)
        df = evaluate_method(
            "Ours",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        print(f"  Avg loss: {df['loss'].mean():.4f}")
        print(f"  Update rate: {(df['decision'] == 'UPDATE').mean():.1%}")

    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        summary = (
            combined_df.groupby("method")
            .agg(
                {
                    "loss": "mean",
                    "cost": "mean",
                }
            )
            .round(4)
        )
        print(summary)

        # Save
        combined_df.to_csv(output_dir / "results.csv", index=False)

        summary_dict = {
            "model_scale": args.model_scale,
            "budget": args.budget,
            "num_batches": args.num_eval_batches,
            "methods": summary.to_dict(),
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)

        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
