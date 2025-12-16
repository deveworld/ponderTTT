# type: ignore
# pyright: reportMissingImports=false
"""
Gradient Norm & TTT Loss Improvement vs Oracle Advantage 상관관계 분석.

목표: "측정 기반" gating의 가능성 검증
- TTT improvement (ttt_loss_step_0 - ttt_loss_step_1): TTT self-supervision에서 얼마나 개선되었는지
- Oracle advantage (loss_skip - loss_update): 실제 output loss 개선량

상관관계가 높으면 → TTT improvement를 gating signal로 사용 가능 (Crawl 단계)
상관관계가 낮으면 → 더 복잡한 probe 필요 (Walk 단계로 진행)

Usage:
    python -m ponderttt.experiments.analyze_gradient_norm --model_scale 125m
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from scipy import stats
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model
from ..utils import cross_entropy_loss
from ..utils.checkpointing import load_checkpoint, unwrap_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between TTT improvement and oracle advantage"
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["125m", "350m", "1b"],
        default="125m",
        help="Model scale",
    )
    parser.add_argument(
        "--ttt_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained TTT checkpoint",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to analyze",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Python",
        help="Programming language for evaluation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (The Stack v2 only has 'train')",
    )
    parser.add_argument(
        "--skip_examples",
        type=int,
        default=10000,
        help="Number of examples to skip (for held-out evaluation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gradient_analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of data loading workers",
    )
    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    return {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]


def get_fast_weight_checksum(model) -> float:
    """Get checksum of fast weights to detect state leakage."""
    w1 = model.fast_layer.W1[...]
    b1 = model.fast_layer.b1[...]
    return float(jnp.sum(w1) + jnp.sum(b1))


@nnx.jit
def compute_skip_and_update_losses(
    model,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
):
    """Compute both SKIP and UPDATE losses, plus TTT stats."""
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

    # TTT internal stats
    ttt_stats = out_update.get("ttt_stats", {})
    ttt_loss_init = ttt_stats.get("ttt_loss_init", jnp.array(0.0))
    ttt_loss_step_0 = ttt_stats.get("ttt_loss_step_0", jnp.array(0.0))
    ttt_loss_step_1 = ttt_stats.get("ttt_loss_step_1", jnp.array(0.0))

    return loss_skip, loss_update, ttt_loss_init, ttt_loss_step_0, ttt_loss_step_1


def main():
    args = parse_args()

    print("=" * 60)
    print("TTT Improvement vs Oracle Advantage Analysis")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Language: {args.language}")
    print(f"Batches: {args.num_batches}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)

    # Load TTT model
    print("\nLoading TTT model...")
    ttt_model, _ = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    # Load checkpoint if provided
    if args.ttt_checkpoint:
        print(f"Loading TTT checkpoint from {args.ttt_checkpoint}...")
        try:
            ckpt = load_checkpoint(args.ttt_checkpoint, target=None)
            if "state" in ckpt and "model" in ckpt["state"]:
                model_state = unwrap_state(ckpt["state"]["model"])
                if "fast_layer" in model_state:
                    nnx.update(ttt_model.fast_layer, model_state["fast_layer"])
                    print("✓ Loaded fast_layer from checkpoint")
                if "fast_norm" in model_state:
                    nnx.update(ttt_model.fast_norm, model_state["fast_norm"])
                    print("✓ Loaded fast_norm from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    # Data iterator
    chunk_size = 512
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=args.split,
        language=args.language,
        batch_size=args.batch_size,
        seq_length=1024,
        chunk_size=chunk_size,
        max_examples=args.batch_size * args.num_batches * 2,
        skip_examples=args.skip_examples,
        num_workers=args.num_workers,
    )

    # Collect data
    print("\nCollecting measurements...")
    all_ttt_improvement = []  # ttt_loss_step_0 - ttt_loss_step_1
    all_ttt_loss_init = []
    all_ttt_loss_step_0 = []
    all_ttt_loss_step_1 = []
    all_oracle_advantage = []  # loss_skip - loss_update
    all_loss_skip = []
    all_loss_update = []

    # State leakage detection
    initial_checksum = get_fast_weight_checksum(ttt_model)
    print(f"[State Check] Initial fast weight checksum: {initial_checksum:.6f}")
    state_leaked = False

    batch_count = 0
    for batch in tqdm(data_iter, total=args.num_batches, desc="Processing"):
        if batch_count >= args.num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        for c_idx in range(num_chunks):
            input_ids = chunks[:, c_idx]
            attention_mask = masks[:, c_idx]
            position_ids = jnp.arange(
                c_idx * chunk_size, (c_idx + 1) * chunk_size, dtype=jnp.int32
            )[None, :].repeat(chunks.shape[0], axis=0)

            # Skip padding-only chunks
            num_valid = jnp.sum(attention_mask[:, 1:])
            if float(num_valid) < 16:
                continue

            # Compute losses
            loss_skip, loss_update, ttt_init, ttt_step0, ttt_step1 = compute_skip_and_update_losses(
                ttt_model,
                input_ids,
                attention_mask,
                position_ids,
            )

            # Store results
            # TTT stats are per-head arrays, take mean
            ttt_init_scalar = float(jnp.mean(ttt_init))
            ttt_step0_scalar = float(jnp.mean(ttt_step0))
            ttt_step1_scalar = float(jnp.mean(ttt_step1))

            all_loss_skip.append(float(loss_skip))
            all_loss_update.append(float(loss_update))
            all_oracle_advantage.append(float(loss_skip - loss_update))
            all_ttt_loss_init.append(ttt_init_scalar)
            all_ttt_loss_step_0.append(ttt_step0_scalar)
            all_ttt_loss_step_1.append(ttt_step1_scalar)
            all_ttt_improvement.append(ttt_step0_scalar - ttt_step1_scalar)

        # Check for state leakage after first batch
        if batch_count == 0:
            checksum_after_batch1 = get_fast_weight_checksum(ttt_model)
            if abs(checksum_after_batch1 - initial_checksum) > 1e-6:
                print("\n⚠️  [STATE LEAKAGE DETECTED] Fast weights changed after batch 1!")
                print(f"    Initial: {initial_checksum:.6f}")
                print(f"    After batch 1: {checksum_after_batch1:.6f}")
                print(f"    Delta: {checksum_after_batch1 - initial_checksum:.6f}")
                state_leaked = True
            else:
                print("\n✓ [State Check] No state leakage after batch 1 (checksum unchanged)")

        batch_count += 1

    # Final state check
    final_checksum = get_fast_weight_checksum(ttt_model)
    if abs(final_checksum - initial_checksum) > 1e-6:
        print("\n⚠️  [STATE LEAKAGE DETECTED] Fast weights changed after all batches!")
        print(f"    Initial: {initial_checksum:.6f}")
        print(f"    Final: {final_checksum:.6f}")
        print(f"    Delta: {final_checksum - initial_checksum:.6f}")
        state_leaked = True
    else:
        print("\n✓ [State Check] No state leakage after all batches (checksum unchanged)")

    if state_leaked:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: State leakage detected!")
        print("Results may be contaminated. TTT forward modifies fast weights.")
        print("Consider adding state reset between batches.")
        print("=" * 60)

    # Convert to numpy
    ttt_improvement = np.array(all_ttt_improvement)
    oracle_advantage = np.array(all_oracle_advantage)
    loss_skip = np.array(all_loss_skip)
    loss_update = np.array(all_loss_update)
    ttt_loss_init = np.array(all_ttt_loss_init)
    ttt_loss_step_0 = np.array(all_ttt_loss_step_0)
    _ttt_loss_step_1 = np.array(all_ttt_loss_step_1)  # noqa: F841

    n_samples = len(oracle_advantage)
    print(f"\nCollected {n_samples} samples")

    # === Correlation Analysis ===
    print("\n" + "=" * 60)
    print("Correlation Analysis")
    print("=" * 60)

    # 1. TTT Improvement vs Oracle Advantage
    pearson_result = stats.pearsonr(ttt_improvement, oracle_advantage)
    spearman_result = stats.spearmanr(ttt_improvement, oracle_advantage)
    pearson_r, pearson_p = pearson_result.statistic, pearson_result.pvalue  # type: ignore[attr-defined]
    spearman_r, spearman_p = spearman_result.statistic, spearman_result.pvalue  # type: ignore[attr-defined]
    print("\n[TTT Improvement vs Oracle Advantage]")
    print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")

    # 2. TTT Loss Step 0 vs Oracle Advantage
    pr2 = stats.pearsonr(ttt_loss_step_0, oracle_advantage)
    sr2 = stats.spearmanr(ttt_loss_step_0, oracle_advantage)
    pearson_r2, spearman_r2 = pr2.statistic, sr2.statistic  # type: ignore[attr-defined]
    print("\n[TTT Loss Step 0 (before update) vs Oracle Advantage]")
    print(f"  Pearson r:  {pearson_r2:.4f}")
    print(f"  Spearman ρ: {spearman_r2:.4f}")

    # 3. TTT Loss Init vs Oracle Advantage
    pr3 = stats.pearsonr(ttt_loss_init, oracle_advantage)
    sr3 = stats.spearmanr(ttt_loss_init, oracle_advantage)
    pearson_r3, spearman_r3 = pr3.statistic, sr3.statistic  # type: ignore[attr-defined]
    print("\n[TTT Loss Init vs Oracle Advantage]")
    print(f"  Pearson r:  {pearson_r3:.4f}")
    print(f"  Spearman ρ: {spearman_r3:.4f}")

    # === 분포 통계 ===
    print("\n" + "=" * 60)
    print("Distribution Statistics")
    print("=" * 60)
    print("\nOracle Advantage (loss_skip - loss_update):")
    print(f"  Mean:   {oracle_advantage.mean():.4f}")
    print(f"  Std:    {oracle_advantage.std():.4f}")
    print(f"  Min:    {oracle_advantage.min():.4f}")
    print(f"  Max:    {oracle_advantage.max():.4f}")
    print(f"  Positive rate: {(oracle_advantage > 0).mean():.2%}")

    print("\nTTT Improvement (step_0 - step_1):")
    print(f"  Mean:   {ttt_improvement.mean():.4f}")
    print(f"  Std:    {ttt_improvement.std():.4f}")
    print(f"  Min:    {ttt_improvement.min():.4f}")
    print(f"  Max:    {ttt_improvement.max():.4f}")
    print(f"  Positive rate: {(ttt_improvement > 0).mean():.2%}")

    print("\nLoss Statistics:")
    print(f"  Loss Skip:   {loss_skip.mean():.4f} ± {loss_skip.std():.4f}")
    print(f"  Loss Update: {loss_update.mean():.4f} ± {loss_update.std():.4f}")

    # === Top-k 예측 정확도 ===
    print("\n" + "=" * 60)
    print("Top-k Prediction Accuracy (using TTT Improvement)")
    print("=" * 60)

    for k in [0.1, 0.2, 0.3, 0.5]:
        # Oracle top-k
        threshold_oracle = np.percentile(oracle_advantage, 100 * (1 - k))
        oracle_topk = oracle_advantage >= threshold_oracle

        # Predicted top-k using TTT improvement
        threshold_pred = np.percentile(ttt_improvement, 100 * (1 - k))
        pred_topk = ttt_improvement >= threshold_pred

        # Overlap
        overlap = (oracle_topk & pred_topk).sum() / oracle_topk.sum()
        print(f"  Top-{k*100:.0f}% overlap: {overlap:.2%}")

    # === 시각화 ===
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. TTT Improvement vs Oracle Advantage (main plot)
    ax = axes[0, 0]
    ax.scatter(ttt_improvement, oracle_advantage, alpha=0.3, s=10)
    ax.set_xlabel("TTT Improvement (step_0 - step_1)")
    ax.set_ylabel("Oracle Advantage (skip - update)")
    ax.set_title(f"TTT Improvement vs Oracle Advantage\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    # Fit line
    z = np.polyfit(ttt_improvement, oracle_advantage, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ttt_improvement.min(), ttt_improvement.max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.5, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
    ax.legend()

    # 2. TTT Loss Step 0 vs Oracle Advantage
    ax = axes[0, 1]
    ax.scatter(ttt_loss_step_0, oracle_advantage, alpha=0.3, s=10)
    ax.set_xlabel("TTT Loss Step 0")
    ax.set_ylabel("Oracle Advantage")
    ax.set_title(f"TTT Loss Step 0 vs Oracle Advantage\nPearson r={pearson_r2:.3f}")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 3. TTT Loss Init vs Oracle Advantage
    ax = axes[0, 2]
    ax.scatter(ttt_loss_init, oracle_advantage, alpha=0.3, s=10)
    ax.set_xlabel("TTT Loss Init")
    ax.set_ylabel("Oracle Advantage")
    ax.set_title(f"TTT Loss Init vs Oracle Advantage\nPearson r={pearson_r3:.3f}")
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. Oracle Advantage histogram
    ax = axes[1, 0]
    ax.hist(oracle_advantage, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', label="Zero")
    ax.axvline(x=oracle_advantage.mean(), color='g', linestyle='--', label=f"Mean={oracle_advantage.mean():.3f}")
    ax.set_xlabel("Oracle Advantage")
    ax.set_ylabel("Count")
    ax.set_title("Oracle Advantage Distribution")
    ax.legend()

    # 5. TTT Improvement histogram
    ax = axes[1, 1]
    ax.hist(ttt_improvement, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', label="Zero")
    ax.axvline(x=ttt_improvement.mean(), color='g', linestyle='--', label=f"Mean={ttt_improvement.mean():.3f}")
    ax.set_xlabel("TTT Improvement")
    ax.set_ylabel("Count")
    ax.set_title("TTT Improvement Distribution")
    ax.legend()

    # 6. Top-k overlap curve
    ax = axes[1, 2]
    k_values = np.arange(0.05, 1.0, 0.05)
    overlaps = []
    for k in k_values:
        threshold_oracle = np.percentile(oracle_advantage, 100 * (1 - k))
        oracle_topk = oracle_advantage >= threshold_oracle
        threshold_pred = np.percentile(ttt_improvement, 100 * (1 - k))
        pred_topk = ttt_improvement >= threshold_pred
        overlap = (oracle_topk & pred_topk).sum() / max(oracle_topk.sum(), 1)
        overlaps.append(overlap)
    ax.plot(k_values * 100, overlaps, 'b-o', markersize=3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Random baseline")
    ax.set_xlabel("Top-k %")
    ax.set_ylabel("Overlap with Oracle")
    ax.set_title("Top-k Prediction Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"correlation_analysis_{args.model_scale}_{args.language}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()

    # === CONCLUSION ===
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if spearman_r > 0.5:
        print(f"✓ Strong correlation (ρ={spearman_r:.3f}) detected!")
        print("  → TTT Improvement can be used as gating signal")
        print("  → Proceed with 'Crawl' phase: threshold-based gating")
    elif spearman_r > 0.3:
        print(f"△ Moderate correlation (ρ={spearman_r:.3f}) detected.")
        print("  → TTT Improvement provides some signal, but not sufficient alone")
        print("  → Consider combining with other signals (entropy, loss ratio)")
    else:
        print(f"✗ Weak correlation (ρ={spearman_r:.3f})")
        print("  → TTT Improvement is NOT a good proxy for oracle advantage")
        print("  → Proceed to 'Walk' phase: LoRA lookahead probe")

    # Save results as JSON
    import json
    results = {
        "model_scale": args.model_scale,
        "language": args.language,
        "n_samples": n_samples,
        "correlations": {
            "ttt_improvement_vs_oracle": {
                "pearson_r": float(pearson_r),
                "spearman_r": float(spearman_r),
            },
            "ttt_loss_step0_vs_oracle": {
                "pearson_r": float(pearson_r2),
                "spearman_r": float(spearman_r2),
            },
            "ttt_loss_init_vs_oracle": {
                "pearson_r": float(pearson_r3),
                "spearman_r": float(spearman_r3),
            },
        },
        "statistics": {
            "oracle_advantage": {
                "mean": float(oracle_advantage.mean()),
                "std": float(oracle_advantage.std()),
                "positive_rate": float((oracle_advantage > 0).mean()),
            },
            "ttt_improvement": {
                "mean": float(ttt_improvement.mean()),
                "std": float(ttt_improvement.std()),
                "positive_rate": float((ttt_improvement > 0).mean()),
            },
        },
    }
    results_path = output_dir / f"correlation_results_{args.model_scale}_{args.language}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
