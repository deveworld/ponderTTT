"""
Toy Experiment: Validate Adaptive TTT Concept

Quick proof-of-concept to verify that adaptive iteration allocation works.
"""

import time

import torch
import torch.nn as nn

from ponderttt.models.adaptive_ttt import HeuristicAdaptiveTTT
from ponderttt.models.ttt_linear import TTTLinearWithStats


def create_synthetic_data(batch_size=4, seq_len=16, hidden_dim=64):
    """
    Create synthetic data with varying difficulty.

    Returns data where some tokens are "easy" (low variance)
    and some are "hard" (high variance).
    """
    # Easy tokens: low variance, high correlation
    easy_tokens = torch.randn(batch_size, seq_len // 2, hidden_dim) * 0.1

    # Hard tokens: high variance, low correlation
    hard_tokens = torch.randn(batch_size, seq_len // 2, hidden_dim) * 1.0

    # Concatenate
    data = torch.cat([easy_tokens, hard_tokens], dim=1)

    return data


def create_logits(x, vocab_size=100):
    """Create mock logits for entropy calculation."""
    # Simple linear projection to vocab_size
    proj = nn.Linear(x.size(-1), vocab_size)
    logits = proj(x)
    return logits


def run_experiment():
    """Run toy experiment comparing fixed vs adaptive TTT."""

    print("=" * 60)
    print("PonderTTT: Toy Experiment")
    print("=" * 60)
    print()

    # Configuration
    batch_size = 4
    seq_len = 16
    hidden_dim = 64
    ttt_dim = 32

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  TTT dim: {ttt_dim}")
    print()

    # Create data
    print("Creating synthetic data...")
    x = create_synthetic_data(batch_size, seq_len, hidden_dim)
    logits = create_logits(x)
    print(f"  Data shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print()

    # ==== Baseline: Fixed TTT ====
    print("-" * 60)
    print("Baseline: Fixed TTT (2 iterations for all tokens)")
    print("-" * 60)

    fixed_ttt = TTTLinearWithStats(
        hidden_dim=hidden_dim,
        ttt_dim=ttt_dim,
        num_iterations=2,
        learning_rate=0.01,
    )

    start_time = time.time()
    output_fixed, stats_fixed = fixed_ttt(x)
    fixed_time = time.time() - start_time

    print(f"  Output shape: {output_fixed.shape}")
    print(f"  Time: {fixed_time:.4f}s")
    print(f"  Final loss: {stats_fixed['final_loss']:.6f}")
    print(f"  Total iterations: {batch_size * seq_len * 2}")
    print()

    # ==== Adaptive TTT: Entropy-based ====
    print("-" * 60)
    print("Adaptive TTT: Entropy-based allocation")
    print("-" * 60)

    adaptive_ttt = HeuristicAdaptiveTTT(
        base_ttt=fixed_ttt,
        difficulty_metric="entropy",
        buckets=[1, 2, 4],
        thresholds=[0.3, 0.7],  # 30% easy, 40% medium, 30% hard
    )

    start_time = time.time()
    output_adaptive, stats_adaptive = adaptive_ttt.forward_adaptive(x, logits=logits)
    adaptive_time = time.time() - start_time

    print(f"  Output shape: {output_adaptive.shape}")
    print(f"  Time: {adaptive_time:.4f}s")

    # Print iteration distribution
    print("\n  Iteration Distribution:")
    for bucket, fraction in stats_adaptive["distribution"].items():
        print(f"    {bucket} iterations: {fraction * 100:.1f}%")

    # Efficiency metrics
    efficiency = stats_adaptive["efficiency"]
    print("\n  Efficiency Metrics:")
    print(f"    Avg iterations: {efficiency['avg_iterations']:.2f}")
    print(f"    Baseline iterations: {efficiency['baseline_iterations']}")
    print(f"    FLOPs ratio: {efficiency['flops_ratio']:.3f}")
    print(f"    FLOPs reduction: {efficiency['flops_reduction'] * 100:.1f}%")

    print(f"\n  Speedup: {fixed_time / adaptive_time:.2f}x")

    # ==== Comparison ====
    print()
    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    output_diff = torch.abs(output_fixed - output_adaptive).mean().item()

    print(f"  Output difference: {output_diff:.6f}")
    print(f"  FLOPs reduction: {efficiency['flops_reduction'] * 100:.1f}%")
    print(f"  Quality maintained: {'✓' if output_diff < 0.1 else '✗'}")

    print()
    print("✓ Experiment completed successfully!")
    print()

    return {
        "fixed": {"output": output_fixed, "stats": stats_fixed, "time": fixed_time},
        "adaptive": {"output": output_adaptive, "stats": stats_adaptive, "time": adaptive_time},
        "comparison": {
            "output_diff": output_diff,
            "flops_reduction": efficiency["flops_reduction"],
            "speedup": fixed_time / adaptive_time,
        },
    }


if __name__ == "__main__":
    # Run experiment
    results = run_experiment()

    print("Next steps:")
    print("  1. Try different difficulty metrics (loss, gradient)")
    print("  2. Tune thresholds for better allocation")
    print("  3. Scale to real language modeling tasks")
