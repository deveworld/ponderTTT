"""
Analyze and visualize WikiText-2 experiment results.

This script loads results from experiments and generates:
- Pareto curves (FLOPs vs Perplexity)
- Allocation distribution histograms
- Per-bucket quality analysis
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def load_all_results(results_dir: str = "experiments/results") -> List[Dict]:
    """Load all experiment results from JSON files."""
    results = []

    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    for filepath in json_files:
        with open(filepath, "r") as f:
            data = json.load(f)
            # Handle both single results and lists of results
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)

    return results


def create_pareto_curve(results: List[Dict], output_dir: str = "experiments/figures"):
    """
    Create Pareto curve showing FLOPs vs Perplexity tradeoff.

    Args:
        results: List of experiment results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    configs = []
    perplexities = []
    flops = []
    labels = []

    for result in results:
        config = result["config"]
        ttt_iters = result.get("ttt_iterations", "adaptive")

        # Skip if missing required fields
        if "test_perplexity" not in result:
            continue

        configs.append(f"{config}-{ttt_iters}")
        perplexities.append(result["test_perplexity"])

        # Estimate FLOPs if not provided
        if "flops_per_token" in result:
            flops.append(result["flops_per_token"])
        else:
            # Estimate based on iterations
            base_flops = 2.5e10
            if config == "adaptive":
                avg_iters = result.get("ttt_stats", {}).get("avg_iterations", 2.0)
                flops.append(base_flops * (avg_iters / 2.0))
            else:
                flops.append(base_flops * (ttt_iters / 2.0))

        if config == "baseline":
            labels.append(f"Fixed-{ttt_iters}")
        else:
            avg_iters = result.get("ttt_stats", {}).get("avg_iterations", "?")
            labels.append(f"Adaptive ({avg_iters:.1f})")

    # Normalize FLOPs (relative to Fixed-4 baseline)
    if flops:
        max_flops = max(flops)
        flops_normalized = [f / max_flops for f in flops]
    else:
        flops_normalized = flops

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot points
    colors = ["#e74c3c" if "Fixed" in label else "#2ecc71" for label in labels]
    sizes = [150 if "Fixed" in label else 200 for label in labels]

    for i, (f, p, label, color, size) in enumerate(
        zip(flops_normalized, perplexities, labels, colors, sizes)
    ):
        plt.scatter(f, p, s=size, c=color, alpha=0.7, edgecolors="black", linewidths=1.5)
        plt.annotate(
            label,
            (f, p),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
        )

    plt.xlabel("Relative FLOPs", fontsize=12, fontweight="bold")
    plt.ylabel("Test Perplexity", fontsize=12, fontweight="bold")
    plt.title("PonderTTT: FLOPs-Perplexity Tradeoff on WikiText-2", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Fixed iterations"),
        Patch(facecolor="#2ecc71", edgecolor="black", label="Adaptive iterations"),
    ]
    plt.legend(handles=legend_elements, loc="best", framealpha=0.9)

    plt.tight_layout()

    # Save
    filepath = os.path.join(output_dir, "pareto_curve_wikitext2.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved Pareto curve to: {filepath}")

    plt.close()


def create_allocation_distribution(results: List[Dict], output_dir: str = "experiments/figures"):
    """
    Create histogram of iteration allocation for adaptive method.

    Args:
        results: List of experiment results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find adaptive results
    adaptive_results = [r for r in results if r["config"] == "adaptive"]

    if not adaptive_results:
        print("No adaptive results found. Skipping allocation distribution plot.")
        return

    # Extract allocation distributions
    for result in adaptive_results:
        if "ttt_stats" not in result or "allocation_distribution" not in result["ttt_stats"]:
            continue

        dist = result["ttt_stats"]["allocation_distribution"]

        # Create figure
        plt.figure(figsize=(8, 6))

        buckets = sorted(dist.keys())
        fractions = [dist[b] for b in buckets]

        bars = plt.bar(
            [str(b) for b in buckets],
            fractions,
            color="#3498db",
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{frac * 100:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xlabel("Number of TTT Iterations", fontsize=12, fontweight="bold")
        plt.ylabel("Fraction of Tokens", fontsize=12, fontweight="bold")
        plt.title("Adaptive TTT: Iteration Allocation Distribution", fontsize=14, fontweight="bold")
        plt.ylim(0, max(fractions) * 1.2)
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save
        filepath = os.path.join(output_dir, "allocation_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved allocation distribution to: {filepath}")

        plt.close()


def create_results_table(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("WikiText-2 Experiment Results")
    print("=" * 80)

    # Filter valid results
    valid_results = [r for r in results if "test_perplexity" in r]

    if not valid_results:
        print("No valid results found.")
        return

    # Sort by perplexity (lower is better)
    results_sorted = sorted(valid_results, key=lambda r: r.get("test_perplexity", float("inf")))

    print(f"{'Config':<20} {'Test PPL':<12} {'FLOPs/token':<15} {'Avg Iters':<12} {'FLOPs ↓':<10}")
    print("-" * 80)

    for result in results_sorted:
        config = result["config"]
        ttt_iters = result.get("ttt_iterations", "adaptive")
        config_name = f"{config}-{ttt_iters}"

        ppl = result["test_perplexity"]
        flops = result.get("flops_per_token", 0)

        if config == "adaptive" and "ttt_stats" in result:
            stats = result["ttt_stats"]
            avg_iters = stats.get("avg_iterations", 0)
            flops_reduction = stats.get("flops_reduction", 0)

            if flops > 0:
                print(
                    f"{config_name:<20} {ppl:<12.2f} {flops:<15.2e} {avg_iters:<12.2f} {flops_reduction * 100:<10.1f}%"
                )
            else:
                print(
                    f"{config_name:<20} {ppl:<12.2f} {'N/A':<15} {avg_iters:<12.2f} {flops_reduction * 100:<10.1f}%"
                )
        else:
            if flops > 0:
                print(f"{config_name:<20} {ppl:<12.2f} {flops:<15.2e} {'-':<12} {'-':<10}")
            else:
                print(f"{config_name:<20} {ppl:<12.2f} {'N/A':<15} {'-':<12} {'-':<10}")

    print("=" * 80 + "\n")


def create_training_curves(results: List[Dict], output_dir: str = "experiments/figures"):
    """
    Create training curves showing validation perplexity over epochs.

    Args:
        results: List of experiment results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for result in results:
        config = result["config"]
        ttt_iters = result.get("ttt_iterations", "adaptive")

        val_ppls = result.get("val_perplexities", [])

        if val_ppls:
            epochs = list(range(1, len(val_ppls) + 1))

            if config == "baseline":
                linestyle = "--"
                color = "#e74c3c" if ttt_iters == 4 else "#f39c12" if ttt_iters == 2 else "#95a5a6"
                label = f"Fixed-{ttt_iters}"
            else:
                linestyle = "-"
                color = "#2ecc71"
                label = "Adaptive"

            plt.plot(
                epochs,
                val_ppls,
                marker="o",
                linestyle=linestyle,
                linewidth=2,
                label=label,
                color=color,
            )

    plt.xlabel("Epoch", fontsize=12, fontweight="bold")
    plt.ylabel("Validation Perplexity", fontsize=12, fontweight="bold")
    plt.title("Training Curves: Validation Perplexity", fontsize=14, fontweight="bold")
    plt.legend(loc="best", framealpha=0.9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filepath = os.path.join(output_dir, "training_curves.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved training curves to: {filepath}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze WikiText-2 experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/figures",
        help="Directory to save figures",
    )

    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} experiment results")

    # Create visualizations
    create_results_table(results)
    create_pareto_curve(results, args.output_dir)
    create_allocation_distribution(results, args.output_dir)
    create_training_curves(results, args.output_dir)

    print("\nAnalysis complete! ✅")


if __name__ == "__main__":
    main()
