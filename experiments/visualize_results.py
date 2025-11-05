"""
Visualization Script for PonderTTT Results

Creates publication-quality plots:
1. Iteration distribution heatmap
2. Correlation scatter plot
3. Quality vs efficiency tradeoff
4. Difficulty distribution by true label
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ponderttt.models.adaptive_ttt import HeuristicAdaptiveTTT
from ponderttt.models.ttt_linear import TTTLinear
from ponderttt.utils.metrics import DifficultyMetrics

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def create_controlled_difficulty_data(batch_size=8, seq_len=64, hidden_dim=128, vocab_size=100):
    """Create synthetic data with controlled difficulty."""
    torch.manual_seed(42)

    total_tokens = batch_size * seq_len
    n_easy = int(total_tokens * 0.3)
    n_medium = int(total_tokens * 0.4)
    n_hard = total_tokens - n_easy - n_medium

    logits_list = []
    difficulty_labels = []

    # Easy tokens
    for _ in range(n_easy):
        logits = torch.randn(vocab_size) * 0.5
        logits[torch.randint(0, vocab_size, (1,))] += 5.0
        logits_list.append(logits)
        difficulty_labels.append(0)

    # Medium tokens
    for _ in range(n_medium):
        logits = torch.randn(vocab_size) * 1.5
        logits_list.append(logits)
        difficulty_labels.append(1)

    # Hard tokens
    for _ in range(n_hard):
        logits = torch.randn(vocab_size) * 0.2
        logits_list.append(logits)
        difficulty_labels.append(2)

    # Shuffle
    indices = torch.randperm(total_tokens)
    logits_tensor = torch.stack(logits_list)[indices]
    difficulty_labels_tensor = torch.tensor(difficulty_labels)[indices]

    # Reshape
    logits_tensor = logits_tensor.view(batch_size, seq_len, vocab_size)
    difficulty_labels_tensor = difficulty_labels_tensor.view(batch_size, seq_len)

    x = torch.randn(batch_size, seq_len, hidden_dim)

    return x, logits_tensor, difficulty_labels_tensor


def plot_iteration_heatmap(iterations, difficulty_scores, save_path="iteration_heatmap.png"):
    """
    Plot heatmap showing iteration allocation across sequence.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Iteration allocation
    im1 = ax1.imshow(iterations.numpy(), aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax1.set_title("Iteration Allocation per Token", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Sequence Position")
    ax1.set_ylabel("Batch")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Iterations", rotation=270, labelpad=20)

    # Plot 2: Difficulty scores
    im2 = ax2.imshow(
        difficulty_scores.numpy(), aspect="auto", cmap="viridis", interpolation="nearest"
    )
    ax2.set_title("Difficulty Scores per Token", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Sequence Position")
    ax2.set_ylabel("Batch")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Difficulty", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_correlation_scatter(difficulty_scores, iterations, save_path="correlation_scatter.png"):
    """
    Scatter plot showing correlation between difficulty and iterations.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten tensors
    difficulty_flat = difficulty_scores.flatten().numpy()
    iterations_flat = iterations.flatten().numpy()

    # Create scatter plot with density coloring
    from scipy.stats import gaussian_kde

    # Calculate point density
    xy = np.vstack([difficulty_flat, iterations_flat])
    z = gaussian_kde(xy)(xy)

    # Sort by density so dense points are plotted last
    idx = z.argsort()
    x, y, z = difficulty_flat[idx], iterations_flat[idx], z[idx]

    scatter = ax.scatter(
        x, y, c=z, s=30, alpha=0.6, cmap="viridis", edgecolors="black", linewidth=0.5
    )

    # Add regression line
    coeffs = np.polyfit(difficulty_flat, iterations_flat, 1)
    poly_fn = np.poly1d(coeffs)
    x_line = np.linspace(difficulty_flat.min(), difficulty_flat.max(), 100)
    ax.plot(
        x_line,
        poly_fn(x_line),
        "r--",
        linewidth=2,
        label=f"Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}",
    )

    # Calculate and display correlation
    correlation = np.corrcoef(difficulty_flat, iterations_flat)[0, 1]

    ax.set_xlabel("Difficulty Score", fontsize=12)
    ax.set_ylabel("Assigned Iterations", fontsize=12)
    ax.set_title(
        f"Difficulty vs Iterations (r = {correlation:.4f})", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Point Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_quality_efficiency_tradeoff(
    quality_results, avg_iters, baseline_iters, save_path="quality_efficiency.png"
):
    """
    Plot quality vs efficiency tradeoff.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Quality per bucket
    buckets = sorted(quality_results.keys())
    mean_diffs = [quality_results[b]["mean_rel_diff"] * 100 for b in buckets]
    counts = [quality_results[b]["count"] for b in buckets]

    colors = ["#2ecc71", "#f39c12", "#e74c3c"][: len(buckets)]
    bars = ax1.bar(
        range(len(buckets)), mean_diffs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count} tokens",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_xlabel("Iteration Bucket", fontsize=12)
    ax1.set_ylabel("Mean Relative Difference (%)", fontsize=12)
    ax1.set_title("Quality Degradation per Bucket", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(buckets)))
    ax1.set_xticklabels([f"{b} iters" for b in buckets])
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Efficiency comparison
    methods = ["Baseline\n(Fixed)", "Adaptive\n(Ours)"]
    iterations = [baseline_iters, avg_iters]
    colors_eff = ["#95a5a6", "#3498db"]

    bars2 = ax2.bar(
        methods, iterations, color=colors_eff, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels
    for bar, val in zip(bars2, iterations):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add efficiency gain annotation
    reduction = (1 - avg_iters / baseline_iters) * 100
    ax2.text(
        0.5,
        max(iterations) * 0.7,
        f"{reduction:.1f}% FLOPs\nReduction",
        ha="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    ax2.set_ylabel("Avg Iterations per Token", fontsize=12)
    ax2.set_title("Computational Efficiency", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, max(iterations) * 1.15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_difficulty_distribution(
    difficulty_scores, true_labels, iterations, save_path="difficulty_distribution.png"
):
    """
    Plot difficulty score distributions by true difficulty level.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = ["Easy", "Medium", "Hard"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    for level in range(3):
        ax = axes[level]
        mask = true_labels == level

        scores = difficulty_scores[mask].flatten().numpy()
        iters = iterations[mask].flatten().numpy()

        # Histogram of difficulty scores
        ax.hist(scores, bins=30, color=colors[level], alpha=0.7, edgecolor="black", linewidth=1)

        # Add statistics
        mean_score = scores.mean()
        mean_iter = iters.mean()

        ax.axvline(
            mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.3f}"
        )

        ax.set_xlabel("Difficulty Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(
            f"{labels[level]} Tokens\n(Avg {mean_iter:.2f} iters)", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_iteration_distribution_pie(distribution, save_path="iteration_pie.png"):
    """
    Pie chart showing iteration distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    buckets = sorted(distribution.keys())
    fractions = [distribution[b] for b in buckets]
    labels = [f"{b} iterations\n({f * 100:.1f}%)" for b, f in zip(buckets, fractions)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"][: len(buckets)]

    _ = ax.pie(
        fractions,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=90,
        explode=[0.05] * len(buckets),
        shadow=True,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title("Iteration Distribution Across Tokens", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("Generating Visualizations for PonderTTT")
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
    baseline_iterations: int = 4

    # Generate data
    print("Generating data...")
    x, logits, true_difficulty_labels = create_controlled_difficulty_data(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
    )

    # Create models
    print("Setting up models...")
    base_ttt = TTTLinear(
        hidden_dim=hidden_dim,
        ttt_dim=ttt_dim,
        num_iterations=4,
    )

    adaptive_ttt = HeuristicAdaptiveTTT(
        base_ttt=base_ttt,
        difficulty_metric="entropy",
        buckets=buckets,
        auto_calibrate=True,
        target_distribution=target_distribution,
    )

    # Run adaptive forward
    print("Running adaptive TTT...")
    adaptive_output, stats = adaptive_ttt.forward_adaptive(x, logits=logits)

    # Get baseline for comparison
    base_ttt.num_iterations = baseline_iterations  # type: ignore[unresolved-attribute]
    baseline_output = base_ttt(x)

    # Extract data
    difficulty_metric = DifficultyMetrics()
    difficulty_scores = difficulty_metric.entropy_based(logits)
    iterations = stats["iterations"]

    # Compute quality per bucket
    quality_results = {}
    for bucket in buckets:
        mask = iterations == bucket
        if mask.sum() == 0:
            continue

        adaptive_tokens = adaptive_output[mask]
        baseline_tokens = baseline_output[mask]
        rel_diff = (
            ((adaptive_tokens - baseline_tokens).abs() / (baseline_tokens.abs() + 1e-8))
            .mean()
            .item()
        )

        quality_results[bucket] = {
            "count": mask.sum().item(),
            "mean_rel_diff": rel_diff,
        }

    avg_iters = iterations.float().mean().item()

    # Create output directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Generate all plots
    print(f"\nGenerating plots in {output_dir}/...")
    print()

    plot_iteration_heatmap(
        iterations, difficulty_scores, save_path=str(output_dir / "iteration_heatmap.png")
    )

    plot_correlation_scatter(
        difficulty_scores, iterations, save_path=str(output_dir / "correlation_scatter.png")
    )

    plot_quality_efficiency_tradeoff(
        quality_results,
        avg_iters,
        baseline_iterations,
        save_path=str(output_dir / "quality_efficiency.png"),
    )

    plot_difficulty_distribution(
        difficulty_scores,
        true_difficulty_labels,
        iterations,
        save_path=str(output_dir / "difficulty_distribution.png"),
    )

    plot_iteration_distribution_pie(
        stats["distribution"], save_path=str(output_dir / "iteration_pie.png")
    )

    print()
    print("=" * 80)
    print(f"✓ All visualizations saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
