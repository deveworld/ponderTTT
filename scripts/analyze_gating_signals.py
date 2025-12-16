#!/usr/bin/env python3
"""
Analyze different gating signals for TTT.

This script examines alternative gating signals beyond ttt_improvement:
1. loss_skip: High skip loss → model struggles → TTT might help
2. ttt_step_0: High initial reconstruction error → potential for improvement

Usage:
    python scripts/analyze_gating_signals.py --model_scale 350m
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ponderttt.models import load_ttt_model
from ponderttt.data import create_data_iterator, get_tokenizer
from ponderttt.utils.checkpointing import load_checkpoint


def analyze_from_csv(csv_path: str):
    """Analyze gating signals from existing detailed_results.csv"""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_path}")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")

    # Filter to relevant columns if they exist
    required_cols = ['ttt_improvement', 'advantage']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns. Available: {df.columns.tolist()}")
        return

    ttt_imp = df['ttt_improvement'].values
    advantage = df['advantage'].values

    print("\n--- Correlation Analysis ---")

    # 1. TTT Improvement vs Advantage (current method)
    r_ttt, p_ttt = stats.pearsonr(ttt_imp, advantage)
    rho_ttt, _ = stats.spearmanr(ttt_imp, advantage)
    print("\n1. TTT Improvement vs Oracle Advantage:")
    print(f"   Pearson r:  {r_ttt:.4f} (p={p_ttt:.2e})")
    print(f"   Spearman ρ: {rho_ttt:.4f}")

    # 2. Check if loss_skip is available
    if 'loss_skip' in df.columns:
        loss_skip = df['loss_skip'].values
        r_skip, p_skip = stats.pearsonr(loss_skip, advantage)
        rho_skip, _ = stats.spearmanr(loss_skip, advantage)
        print("\n2. Loss Skip vs Oracle Advantage:")
        print(f"   Pearson r:  {r_skip:.4f} (p={p_skip:.2e})")
        print(f"   Spearman ρ: {rho_skip:.4f}")

    # 3. Simulate different gating strategies
    print("\n--- Gating Strategy Simulation ---")
    print("Target update rate: 50%")

    # Get loss values if available
    if 'loss_skip' not in df.columns:
        print("loss_skip column not available for simulation")
        return

    loss_skip = df['loss_skip'].values
    loss_update = df['loss_update'].values if 'loss_update' in df.columns else None

    # Strategy 1: TTT Improvement (current)
    threshold_ttt = np.median(ttt_imp)
    decisions_ttt = ttt_imp > threshold_ttt

    # Strategy 2: Loss Skip (high loss → update)
    threshold_skip = np.median(loss_skip)
    decisions_skip = loss_skip > threshold_skip

    # Strategy 3: Oracle (positive advantage → update)
    threshold_oracle = np.median(advantage)
    decisions_oracle = advantage > threshold_oracle

    # Strategy 4: Random
    np.random.seed(42)
    decisions_random = np.random.random(len(advantage)) > 0.5

    def compute_loss(decisions, loss_skip, loss_update):
        """Compute average loss given decisions"""
        losses = np.where(decisions, loss_update, loss_skip)
        return losses.mean()

    if loss_update is not None:
        print("\nAverage Loss by Strategy:")
        print(f"  Oracle (upper bound):     {compute_loss(decisions_oracle, loss_skip, loss_update):.4f}")
        print(f"  TTT Improvement:          {compute_loss(decisions_ttt, loss_skip, loss_update):.4f}")
        print(f"  Loss Skip (high→update):  {compute_loss(decisions_skip, loss_skip, loss_update):.4f}")
        print(f"  Random:                   {compute_loss(decisions_random, loss_skip, loss_update):.4f}")
        print(f"  Always Skip:              {loss_skip.mean():.4f}")
        print(f"  Always Update:            {loss_update.mean():.4f}")

    # Decision overlap with oracle
    print("\nDecision Overlap with Oracle:")
    print(f"  TTT Improvement:          {np.mean(decisions_ttt == decisions_oracle):.2%}")
    print(f"  Loss Skip (high→update):  {np.mean(decisions_skip == decisions_oracle):.2%}")
    print(f"  Random:                   {np.mean(decisions_random == decisions_oracle):.2%}")


def collect_and_analyze(model_scale: str, checkpoint_path: str, num_batches: int = 100):
    """Collect fresh data and analyze gating signals"""
    print(f"\n{'='*60}")
    print(f"Collecting data for {model_scale}")
    print(f"{'='*60}")

    # Model name mapping
    model_map = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}
    model_name = model_map[model_scale]

    # Load model
    print(f"Loading model: {model_name}")
    model, config = load_ttt_model(model_name, seed=42)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    restored_state = load_checkpoint(checkpoint_path)
    nnx.update(model, restored_state)

    # Restore exp_decay_weight
    for name, module in model.iter_modules():
        if hasattr(module, 'exp_decay_weight') and hasattr(module, 'config'):
            eta_decay_rate = module.config.eta_decay_rate
            mini_batch_size = module.config.mini_batch_size
            position_offset = jnp.arange(mini_batch_size) - (mini_batch_size - 1)
            module.exp_decay_weight = jnp.exp(eta_decay_rate * position_offset.astype(jnp.float32))
            print(f"  Applied eta_decay_rate={eta_decay_rate}")

    # Setup data
    tokenizer = get_tokenizer()
    data_iter = create_data_iterator(
        language="Python",
        split="train",
        tokenizer=tokenizer,
        seq_length=256,
        batch_size=1,
        skip_examples=160000,  # Use held-out data
        seed=42,
    )

    # Define forward function
    @nnx.jit
    def forward_with_stats(model, input_ids, attention_mask, position_ids):
        # SKIP path
        out_skip = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=False)
        logits_skip = out_skip["logits"][:, :-1]
        targets = input_ids[:, 1:]
        loss_skip = -jnp.sum(
            jax.nn.log_softmax(logits_skip, axis=-1) * jax.nn.one_hot(targets, logits_skip.shape[-1]),
            axis=-1
        ).mean()

        # UPDATE path
        out_update = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_ttt=True)
        logits_update = out_update["logits"][:, :-1]
        loss_update = -jnp.sum(
            jax.nn.log_softmax(logits_update, axis=-1) * jax.nn.one_hot(targets, logits_update.shape[-1]),
            axis=-1
        ).mean()

        ttt_stats = out_update.get("ttt_stats", {})
        ttt_loss_step_0 = ttt_stats.get("ttt_loss_step_0", jnp.array(0.0))
        ttt_loss_step_1 = ttt_stats.get("ttt_loss_step_1", jnp.array(0.0))

        return loss_skip, loss_update, ttt_loss_step_0, ttt_loss_step_1

    # Collect data
    results = {
        "loss_skip": [],
        "loss_update": [],
        "ttt_step_0": [],
        "ttt_step_1": [],
        "ttt_improvement": [],
        "advantage": [],
    }

    print(f"\nCollecting {num_batches} batches...")
    for batch_idx, batch in enumerate(data_iter):
        if batch_idx >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        for c_idx in range(num_chunks):
            input_ids = chunks[:, c_idx]
            attention_mask = masks[:, c_idx]
            chunk_len = input_ids.shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = position_ids + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, input_ids.shape)

            loss_skip, loss_update, ttt_0, ttt_1 = forward_with_stats(
                model, input_ids, attention_mask, position_ids
            )

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            ttt_0_val = float(ttt_0)
            ttt_1_val = float(ttt_1)

            results["loss_skip"].append(loss_skip_val)
            results["loss_update"].append(loss_update_val)
            results["ttt_step_0"].append(ttt_0_val)
            results["ttt_step_1"].append(ttt_1_val)
            results["ttt_improvement"].append(ttt_0_val - ttt_1_val)
            results["advantage"].append(loss_skip_val - loss_update_val)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Save and analyze
    df = pd.DataFrame(results)
    output_path = f"outputs/gating_analysis_{model_scale}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Run analysis
    analyze_from_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_scale", type=str, default="350m", choices=["125m", "350m"])
    parser.add_argument("--csv", type=str, help="Path to existing detailed_results.csv")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for fresh collection")
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()

    if args.csv:
        analyze_from_csv(args.csv)
    elif args.checkpoint:
        collect_and_analyze(args.model_scale, args.checkpoint, args.num_batches)
    else:
        # Try to find existing results
        default_paths = [
            f"outputs/eval/{args.model_scale}_python/detailed_results.csv",
            f"outputs/gating_analysis_{args.model_scale}.csv",
        ]
        for path in default_paths:
            if Path(path).exists():
                analyze_from_csv(path)
                return

        print("No existing results found. Please provide --csv or --checkpoint")
        print(f"Searched: {default_paths}")


if __name__ == "__main__":
    main()
