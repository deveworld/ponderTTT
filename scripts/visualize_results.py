"""
Visualize training results and create plots.

Usage:
    python scripts/visualize_results.py --results_file outputs/policy/125m/training_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("colorblind")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize PonderTTT results")

    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure format",
    )

    return parser.parse_args()


def load_results(results_file: str) -> Dict:
    """Load results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def plot_training_history(
    results: Dict,
    output_dir: Path,
    format: str = "png",
):
    """Plot training history (cost, reward, lambda)."""
    history = results.get("training_history", [])

    if not history:
        print("No training history found.")
        return

    iterations = [h["iteration"] for h in history]
    avg_costs = [h["avg_cost"] for h in history]
    avg_rewards = [h["avg_reward"] for h in history]
    lambdas = [h["lambda"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Average cost
    axes[0].plot(iterations, avg_costs, marker='o', linewidth=2)
    if "config" in results and "budget_limit" in results["config"]:
        axes[0].axhline(
            y=results["config"]["budget_limit"],
            color='r',
            linestyle='--',
            label="Budget limit",
        )
        axes[0].legend()
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Average Cost (x)")
    axes[0].set_title("Cost Over Training")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Average reward
    axes[1].plot(iterations, avg_rewards, marker='s', linewidth=2, color='green')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Reward Over Training")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Lambda (Lagrangian multiplier)
    axes[2].plot(iterations, lambdas, marker='^', linewidth=2, color='orange')
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Lambda (Penalty)")
    axes[2].set_title("PID Lagrangian Multiplier")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"training_history.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_action_distribution(
    results: Dict,
    output_dir: Path,
    format: str = "png",
):
    """Plot action distribution over time."""
    if "chunks" not in results:
        print("No chunk-level data found.")
        return

    chunks = results["chunks"]
    actions = [c["action"] for c in chunks]

    # Count action frequencies
    action_names = ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    action_counts = {name: actions.count(name) for name in action_names}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Bar chart
    axes[0].bar(action_counts.keys(), action_counts.values())
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Action Distribution")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add percentages on bars
    total = sum(action_counts.values())
    if total > 0:
        for i, (action, count) in enumerate(action_counts.items()):
            percentage = 100 * count / total
            axes[0].text(i, count, f"{percentage:.1f}%", ha='center', va='bottom')

    # Plot 2: Pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[1].pie(
        action_counts.values(),
        labels=action_counts.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
    )
    axes[1].set_title("Action Distribution")

    plt.tight_layout()
    output_file = output_dir / f"action_distribution.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_cost_vs_quality(
    results: Dict,
    output_dir: Path,
    format: str = "png",
):
    """Plot cost vs quality trade-off."""
    if "chunks" not in results:
        print("No chunk-level data found.")
        return

    chunks = results["chunks"]
    costs = [c["cost"] for c in chunks]
    losses = [c["loss"] for c in chunks]

    plt.figure(figsize=(8, 6))
    plt.scatter(costs, losses, alpha=0.5)
    plt.xlabel("Computational Cost (x)")
    plt.ylabel("Loss")
    plt.title("Cost vs Quality Trade-off")
    plt.grid(True, alpha=0.3)

    # Add trend line
    if len(costs) >= 2 and not np.allclose(costs, costs[0]):
        z = np.polyfit(costs, losses, 1)
        p = np.poly1d(z)
        plt.plot(costs, p(costs), "r--", alpha=0.8, label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")
        plt.legend()

    plt.tight_layout()
    output_file = output_dir / f"cost_vs_quality.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_summary_stats(
    results: Dict,
    output_dir: Path,
    format: str = "png",
):
    """Plot summary statistics."""
    if "summary" not in results:
        print("No summary statistics found.")
        return

    summary = results["summary"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Create text summary
    def _fmt_float(value, fmt):
        if isinstance(value, (int, float)):
            return fmt.format(value)
        return str(value)

    text_lines = [
        "Training Summary",
        "=" * 40,
        "",
        f"Total Chunks: {summary.get('total_chunks', 'N/A')}",
        f"Average Loss: {_fmt_float(summary.get('avg_loss', 'N/A'), '{:.4f}')}",
        f"Average Cost: {_fmt_float(summary.get('avg_cost', 'N/A'), '{:.2f}x')}",
        f"Total Cost: {_fmt_float(summary.get('total_cost', 'N/A'), '{:.2f}x')}",
        "",
    ]

    if "config" in results:
        text_lines.append("Configuration:")
        text_lines.append("-" * 40)
        for key, value in results["config"].items():
            text_lines.append(f"  {key}: {value}")

    text = "\n".join(text_lines)
    ax.text(
        0.1, 0.9,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    output_file = output_dir / f"summary.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main visualization function."""
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Results Visualization")
    print("=" * 60)
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load results
    try:
        results = load_results(args.results_file)
        print(f" Loaded results from {args.results_file}")
    except Exception as e:
        print(f"[FAIL] Failed to load results: {e}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    plot_training_history(results, output_dir, args.format)
    plot_action_distribution(results, output_dir, args.format)
    plot_cost_vs_quality(results, output_dir, args.format)
    plot_summary_stats(results, output_dir, args.format)

    print("\n" + "=" * 60)
    print(" Visualization complete!")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
