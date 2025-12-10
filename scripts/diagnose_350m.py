"""Diagnostic script to isolate 350M TTT correlation issue."""

import argparse
import numpy as np
import jax.numpy as jnp
from scipy import stats as scipy_stats

from ponderttt.models import load_ttt_model, TTTTransformerLM
from ponderttt.data import create_data_iterator, get_tokenizer


def run_diagnostic(
    model_name: str,
    language: str,
    num_batches: int = 100,
    batch_size: int = 4,
    chunk_size: int = 512,
    seq_length: int = 512,
    use_checkpoint: str | None = None,
):
    """Run diagnostic to measure TTT improvement vs oracle correlation."""
    print(f"\n{'='*60}")
    print(f"Diagnostic: {model_name} on {language}")
    print(f"Checkpoint: {use_checkpoint or 'FRESH (no checkpoint)'}")
    print('='*60)

    # Load model
    if use_checkpoint:
        from ponderttt.utils.checkpointing import load_checkpoint, unwrap_state
        model, config = load_ttt_model(model_name=model_name, seed=42)
        ckpt = load_checkpoint(use_checkpoint, target=None)
        if ckpt and "fast_layer" in ckpt:
            fast_layer_state = unwrap_state(ckpt["fast_layer"])
            model.fast_layer.update(fast_layer_state)
            print(f"  ✓ Loaded fast_layer from checkpoint")
        if ckpt and "fast_norm" in ckpt:
            fast_norm_state = unwrap_state(ckpt["fast_norm"])
            model.fast_norm.update(fast_norm_state)
            print(f"  ✓ Loaded fast_norm from checkpoint")
        print(f"Loaded checkpoint from {use_checkpoint}")
    else:
        model, config = load_ttt_model(model_name=model_name, seed=42)
        print("Using FRESH model (no checkpoint)")

    assert isinstance(model, TTTTransformerLM)

    # Print TTT config
    print(f"\nTTT Config:")
    print(f"  hidden_dim: {model.ttt_config.hidden_dim}")
    print(f"  num_heads: {model.ttt_config.num_heads}")
    print(f"  head_dim: {model.ttt_config.head_dim}")
    print(f"  hidden_dim / 768 = {model.ttt_config.hidden_dim / 768:.4f}")

    # Load data
    tokenizer = get_tokenizer("gpt2")
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        language=language,
        batch_size=batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        num_workers=8,
    )

    # Collect metrics
    all_ttt_improvements = []
    all_advantages = []
    all_ttt_step0 = []
    all_ttt_step1 = []

    print(f"\nRunning evaluation on {num_batches} batches...")

    for batch_idx, batch in enumerate(data_iter):
        if batch_idx >= num_batches:
            break

        input_ids = jnp.array(batch["input_ids"])

        # SKIP output (no TTT)
        skip_output = model(input_ids, use_ttt=False)
        loss_skip = float(jnp.mean(skip_output["loss"]))

        # UPDATE output (with TTT)
        update_output = model(input_ids, use_ttt=True)
        loss_update = float(jnp.mean(update_output["loss"]))
        ttt_stats = update_output.get("ttt_stats", {})

        # TTT improvement
        ttt_step_0 = float(jnp.mean(ttt_stats["ttt_loss_step_0"]))
        ttt_step_1 = float(jnp.mean(ttt_stats["ttt_loss_step_1"]))
        ttt_improvement = ttt_step_0 - ttt_step_1

        # Oracle advantage
        advantage = loss_skip - loss_update

        all_ttt_improvements.append(ttt_improvement)
        all_advantages.append(advantage)
        all_ttt_step0.append(ttt_step_0)
        all_ttt_step1.append(ttt_step_1)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}")

    # Compute correlation
    ttt_arr = np.array(all_ttt_improvements)
    adv_arr = np.array(all_advantages)

    pearson_r, _ = scipy_stats.pearsonr(ttt_arr, adv_arr)
    spearman_r, _ = scipy_stats.spearmanr(ttt_arr, adv_arr)

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)

    print(f"\nTTT Loss Statistics:")
    print(f"  Mean step_0: {np.mean(all_ttt_step0):.6f}")
    print(f"  Mean step_1: {np.mean(all_ttt_step1):.6f}")
    print(f"  Mean improvement: {np.mean(ttt_arr):.6f}")
    print(f"  Std improvement: {np.std(ttt_arr):.6f}")

    print(f"\nOracle Advantage Statistics:")
    print(f"  Mean advantage: {np.mean(adv_arr):.6f}")
    print(f"  Std advantage: {np.std(adv_arr):.6f}")

    print(f"\nCorrelation (TTT Improvement vs Oracle Advantage):")
    print(f"  Pearson r:  {pearson_r:.4f}")
    print(f"  Spearman ρ: {spearman_r:.4f}")

    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print('='*60)

    if spearman_r > 0.3:
        print("✓ GOOD: Positive correlation - TTT improvement predicts oracle")
    elif spearman_r > 0:
        print("⚠ WEAK: Slightly positive correlation - signal is noisy")
    elif spearman_r > -0.3:
        print("⚠ BAD: Near-zero or slightly negative correlation")
    else:
        print("✗ VERY BAD: Strong negative correlation - signal is INVERTED!")
        print("  This means TTT reconstruction improvement ANTI-correlates with main task benefit")

    return {
        "model_name": model_name,
        "language": language,
        "checkpoint": use_checkpoint,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "mean_improvement": float(np.mean(ttt_arr)),
        "mean_advantage": float(np.mean(adv_arr)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium", help="Model name (gpt2, gpt2-medium)")
    parser.add_argument("--language", default="JavaScript", help="Language for OOD test")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length (must be <= 1024 for GPT-2)")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (optional)")
    args = parser.parse_args()

    # Run diagnostic
    result = run_diagnostic(
        model_name=args.model,
        language=args.language,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        chunk_size=args.chunk_size,
        use_checkpoint=args.checkpoint,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Model: {result['model_name']}")
    print(f"Language: {result['language']}")
    print(f"Checkpoint: {result['checkpoint'] or 'FRESH'}")
    print(f"Spearman ρ: {result['spearman_r']:.4f}")
