"""
Day 3 Analysis: Baseline Adjustment, Quality Analysis, and Correlation Study

This experiment:
1. Compares adaptive TTT against baseline=4 (proper comparison)
2. Analyzes quality per difficulty bucket
3. Computes correlation between difficulty and iterations
4. Provides comprehensive statistics
"""

from typing import Any, Dict

import torch

from ponderttt.models.adaptive_ttt import HeuristicAdaptiveTTT
from ponderttt.models.ttt_linear import TTTLinear
from ponderttt.utils.metrics import DifficultyMetrics


def create_controlled_difficulty_data(batch_size=8, seq_len=64, hidden_dim=128, vocab_size=100):
    """
    Create synthetic data with three distinct difficulty levels.

    Returns:
        x: Input tensor
        logits: Logits for difficulty computation
        true_difficulty_labels: Ground truth difficulty (0=easy, 1=medium, 2=hard)
    """
    torch.manual_seed(42)

    # Calculate tokens per difficulty level (30% easy, 40% medium, 30% hard)
    total_tokens = batch_size * seq_len
    n_easy = int(total_tokens * 0.3)
    n_medium = int(total_tokens * 0.4)
    n_hard = total_tokens - n_easy - n_medium

    # Generate logits with different entropy levels
    logits_list = []
    difficulty_labels = []

    # Easy tokens: Low entropy (peaked distribution)
    for _ in range(n_easy):
        logits = torch.randn(vocab_size) * 0.5  # Low variance
        logits[torch.randint(0, vocab_size, (1,))] += 5.0  # One dominant class
        logits_list.append(logits)
        difficulty_labels.append(0)

    # Medium tokens: Medium entropy
    for _ in range(n_medium):
        logits = torch.randn(vocab_size) * 1.5  # Medium variance
        logits_list.append(logits)
        difficulty_labels.append(1)

    # Hard tokens: High entropy (nearly uniform)
    for _ in range(n_hard):
        logits = torch.randn(vocab_size) * 0.2  # Very uniform
        logits_list.append(logits)
        difficulty_labels.append(2)

    # Shuffle to mix difficulties
    indices = torch.randperm(total_tokens)
    logits_tensor = torch.stack(logits_list)[indices]
    difficulty_labels_tensor = torch.tensor(difficulty_labels)[indices]

    # Reshape
    logits_tensor = logits_tensor.view(batch_size, seq_len, vocab_size)
    difficulty_labels_tensor = difficulty_labels_tensor.view(batch_size, seq_len)

    # Generate input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    return x, logits_tensor, difficulty_labels_tensor


def compute_correlation(iterations, difficulties):
    """Compute Pearson correlation coefficient."""
    iterations = iterations.flatten().float()
    difficulties = difficulties.flatten().float()

    # Remove NaN values
    mask = ~(torch.isnan(iterations) | torch.isnan(difficulties))
    iterations = iterations[mask]
    difficulties = difficulties[mask]

    if len(iterations) < 2:
        return 0.0

    # Compute Pearson correlation
    mean_iter = iterations.mean()
    mean_diff = difficulties.mean()

    numerator = ((iterations - mean_iter) * (difficulties - mean_diff)).sum()
    denominator = torch.sqrt(
        ((iterations - mean_iter) ** 2).sum() * ((difficulties - mean_diff) ** 2).sum()
    )

    if denominator == 0:
        return 0.0

    correlation = (numerator / denominator).item()
    return correlation


def analyze_quality_per_bucket(
    outputs_by_bucket, baseline_output, iteration_assignments, bucket_values
):
    """
    Analyze output quality for each difficulty bucket.

    Returns:
        Dictionary with quality metrics per bucket
    """
    results = {}

    for bucket_idx, bucket_iter in enumerate(bucket_values):
        # Find tokens assigned to this bucket
        mask = iteration_assignments == bucket_iter

        if mask.sum() == 0:
            continue

        # Get outputs for this bucket
        adaptive_tokens = outputs_by_bucket[mask]
        baseline_tokens = baseline_output[mask]

        # Compute quality metrics
        abs_diff = (adaptive_tokens - baseline_tokens).abs()
        rel_diff = abs_diff / (baseline_tokens.abs() + 1e-8)

        results[bucket_iter] = {
            "count": mask.sum().item(),
            "percentage": (mask.sum().float() / mask.numel() * 100).item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "mean_rel_diff": rel_diff.mean().item(),
            "max_abs_diff": abs_diff.max().item(),
        }

    return results


def run_experiment():
    """Run comprehensive Day 3 analysis."""

    print("=" * 80)
    print("Day 3 Analysis: Baseline Adjustment & Quality Study")
    print("=" * 80)
    print()

    # Configuration
    batch_size: int = 8
    seq_len: int = 64
    hidden_dim: int = 128
    ttt_dim: int = 64
    vocab_size: int = 100
    buckets: list[int] = [1, 2, 4]
    target_distribution: list[float] = [0.3, 0.4, 0.3]
    baseline_iterations: int = 4  # Changed from 2!

    config: Dict[str, Any] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "ttt_dim": ttt_dim,
        "vocab_size": vocab_size,
        "buckets": buckets,
        "target_distribution": target_distribution,
        "baseline_iterations": baseline_iterations,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Generate controlled difficulty data
    print("Generating controlled difficulty data...")
    x, logits, true_difficulty_labels = create_controlled_difficulty_data(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
    )
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  True difficulty distribution: {torch.bincount(true_difficulty_labels.flatten())}")
    print()

    # Create base TTT layer
    base_ttt = TTTLinear(
        hidden_dim=hidden_dim,
        ttt_dim=ttt_dim,
        num_iterations=2,  # Will be overridden
    )

    # ==========================================
    # Experiment 1: Baseline (Fixed 4 iterations)
    # ==========================================
    print("-" * 80)
    print("Experiment 1: Baseline (Fixed 4 iterations)")
    print("-" * 80)

    base_ttt.num_iterations = baseline_iterations  # type: ignore[unresolved-attribute]
    baseline_output = base_ttt(x)

    total_tokens = batch_size * seq_len
    baseline_steps = total_tokens * baseline_iterations

    print(f"  Total tokens: {total_tokens}")
    print(f"  Iterations per token: {baseline_iterations}")
    print(f"  Total gradient steps: {baseline_steps}")
    print()

    # ==========================================
    # Experiment 2: Adaptive TTT
    # ==========================================
    print("-" * 80)
    print("Experiment 2: Adaptive TTT (Auto-Calibrated)")
    print("-" * 80)

    # Create adaptive wrapper
    adaptive_ttt = HeuristicAdaptiveTTT(
        base_ttt=base_ttt,
        difficulty_metric="entropy",
        buckets=buckets,
        thresholds=None,  # Will be auto-calibrated
        auto_calibrate=True,
        target_distribution=target_distribution,
    )

    # Run adaptive forward pass
    adaptive_output, stats = adaptive_ttt.forward_adaptive(x, logits=logits)

    # Display results
    print("Calibration:")
    if adaptive_ttt.allocator.calibrated and adaptive_ttt.allocator.thresholds:
        print(f"  Thresholds: {[f'{t:.4f}' for t in adaptive_ttt.allocator.thresholds]}")

    print("\nIteration Distribution:")
    for bucket, fraction in stats["distribution"].items():
        count = int(fraction * total_tokens)
        pct = fraction * 100
        print(f"  {bucket} iterations: {count:4d} tokens ({pct:5.1f}%)")

    print("\nEfficiency Metrics:")
    total_iters = sum(
        bucket * fraction * total_tokens for bucket, fraction in stats["distribution"].items()
    )
    avg_iters = total_iters / total_tokens
    flops_ratio = avg_iters / baseline_iterations
    flops_reduction = (1 - flops_ratio) * 100

    print(f"  Total gradient steps: {total_iters}")
    print(f"  Avg iterations per token: {avg_iters:.3f}")
    print(f"  FLOPs vs baseline: {flops_ratio:.3f}x")
    print(f"  FLOPs reduction: {flops_reduction:.1f}%")
    print()

    # ==========================================
    # Analysis 1: Quality per Bucket
    # ==========================================
    print("-" * 80)
    print("Analysis 1: Quality per Difficulty Bucket")
    print("-" * 80)

    # Get iteration assignments
    difficulty_metric = DifficultyMetrics()
    difficulty_scores = difficulty_metric.entropy_based(logits)
    iteration_assignments = adaptive_ttt.allocator.allocate(difficulty_scores)

    quality_results = analyze_quality_per_bucket(
        adaptive_output, baseline_output, iteration_assignments, buckets
    )

    for bucket_iter in sorted(quality_results.keys()):
        result = quality_results[bucket_iter]
        print(f"\nBucket: {bucket_iter} iterations")
        print(f"  Token count: {result['count']} ({result['percentage']:.1f}%)")
        print(f"  Mean absolute diff: {result['mean_abs_diff']:.6f}")
        print(f"  Mean relative diff: {result['mean_rel_diff'] * 100:.2f}%")
        print(f"  Max absolute diff: {result['max_abs_diff']:.6f}")

    # Overall quality
    overall_abs_diff = (adaptive_output - baseline_output).abs().mean().item()
    overall_rel_diff = (
        ((adaptive_output - baseline_output).abs() / (baseline_output.abs() + 1e-8)).mean().item()
    )

    print("\nOverall Quality:")
    print(f"  Mean absolute diff: {overall_abs_diff:.6f}")
    print(f"  Mean relative diff: {overall_rel_diff * 100:.2f}%")
    print()

    # ==========================================
    # Analysis 2: Correlation Study
    # ==========================================
    print("-" * 80)
    print("Analysis 2: Correlation Between Difficulty and Iterations")
    print("-" * 80)

    # Compute correlation
    correlation = compute_correlation(iteration_assignments.float(), difficulty_scores)

    print(f"Pearson correlation (iterations vs difficulty): {correlation:.4f}")

    # Interpretation
    if correlation > 0.5:
        print("  Interpretation: Strong positive correlation ✓")
    elif correlation > 0.3:
        print("  Interpretation: Moderate positive correlation ✓")
    elif correlation > 0.1:
        print("  Interpretation: Weak positive correlation ⚠️")
    else:
        print("  Interpretation: No clear correlation ✗")

    print()

    # Additional correlation with true difficulty labels
    correlation_true = compute_correlation(
        iteration_assignments.float(), true_difficulty_labels.float()
    )
    print(f"Pearson correlation (iterations vs true difficulty): {correlation_true:.4f}")
    print()

    # ==========================================
    # Analysis 3: Difficulty Distribution
    # ==========================================
    print("-" * 80)
    print("Analysis 3: Difficulty Score Distribution")
    print("-" * 80)

    # Statistics per true difficulty level
    for level in range(3):
        mask = true_difficulty_labels == level
        if mask.sum() == 0:
            continue

        level_name = ["Easy", "Medium", "Hard"][level]
        scores = difficulty_scores[mask]
        iters = iteration_assignments[mask].float()

        print(f"\n{level_name} tokens (true difficulty={level}):")
        print(f"  Count: {mask.sum().item()}")
        print(f"  Difficulty score: mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"  Assigned iterations: mean={iters.mean():.2f}, std={iters.std():.2f}")

        # Distribution of assigned iterations
        for bucket in buckets:
            count = (iters == bucket).sum().item()
            pct = count / mask.sum().item() * 100
            print(f"    {bucket} iters: {count:3d} ({pct:5.1f}%)")

    print()

    # ==========================================
    # Summary
    # ==========================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✓ Baseline Adjustment:")
    print(f"  - New baseline: {baseline_iterations} iterations")
    print(f"  - Adaptive average: {avg_iters:.3f} iterations")
    print(f"  - FLOPs reduction: {flops_reduction:.1f}%")

    if flops_reduction > 30:
        print(f"  - Status: ✓ EXCELLENT (>{30}%)")
    elif flops_reduction > 20:
        print(f"  - Status: ✓ GOOD (>{20}%)")
    elif flops_reduction > 10:
        print(f"  - Status: ⚠️ MODERATE (>{10}%)")
    else:
        print(f"  - Status: ✗ INSUFFICIENT (<{10}%)")

    print("\n✓ Quality Analysis:")
    print(f"  - Mean relative difference: {overall_rel_diff * 100:.2f}%")

    if overall_rel_diff < 0.01:
        print("  - Status: ✓ EXCELLENT (<1%)")
    elif overall_rel_diff < 0.05:
        print("  - Status: ✓ GOOD (<5%)")
    elif overall_rel_diff < 0.10:
        print("  - Status: ⚠️ ACCEPTABLE (<10%)")
    else:
        print("  - Status: ✗ POOR (>10%)")

    print("\n✓ Correlation Analysis:")
    print(f"  - Difficulty vs iterations: r={correlation:.4f}")

    if correlation > 0.3:
        print("  - Status: ✓ PASS (r > 0.3)")
    else:
        print("  - Status: ✗ FAIL (r < 0.3)")

    # Overall assessment
    print(f"\n{'=' * 80}")

    success_criteria = {
        "efficiency": flops_reduction > 20,
        "quality": overall_rel_diff < 0.10,
        "correlation": correlation > 0.3,
    }

    all_passed = all(success_criteria.values())

    if all_passed:
        print("Overall Status: ✓ ALL CRITERIA MET - READY TO PROCEED")
    else:
        print("Overall Status: ⚠️ SOME CRITERIA NOT MET - NEEDS TUNING")
        for criterion, passed in success_criteria.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {criterion}")

    print(f"{'=' * 80}\n")

    return {
        "efficiency": flops_reduction,
        "quality": overall_rel_diff,
        "correlation": correlation,
        "success": all_passed,
        "quality_per_bucket": quality_results,
    }


if __name__ == "__main__":
    results = run_experiment()
