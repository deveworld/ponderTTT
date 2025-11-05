"""
Generate demo results to test visualization pipeline.

Creates realistic-looking results based on Phase 1 expectations.
"""

import json
import os
from typing import Any, Dict, List


def generate_demo_results() -> None:
    """Generate demo results matching expected outcomes."""
    results: List[Dict[str, Any]] = []

    # Fixed-1: Fast but poor quality
    results.append(
        {
            "config": "baseline",
            "ttt_iterations": 1,
            "num_params": 44122112,
            "train_losses": [4.82, 4.65, 4.58],
            "val_perplexities": [126.3, 118.5, 115.2],
            "test_loss": 4.75,
            "test_perplexity": 115.5,
            "flops_per_token": 2.1e10,
        }
    )

    # Fixed-2: Balanced
    results.append(
        {
            "config": "baseline",
            "ttt_iterations": 2,
            "num_params": 44122112,
            "train_losses": [4.78, 4.58, 4.48],
            "val_perplexities": [119.2, 105.8, 98.6],
            "test_loss": 4.58,
            "test_perplexity": 97.5,
            "flops_per_token": 2.45e10,
        }
    )

    # Fixed-4: Best quality but slow
    results.append(
        {
            "config": "baseline",
            "ttt_iterations": 4,
            "num_params": 44122112,
            "train_losses": [4.75, 4.52, 4.42],
            "val_perplexities": [116.8, 98.3, 95.2],
            "test_loss": 4.55,
            "test_perplexity": 94.6,
            "flops_per_token": 3.15e10,
        }
    )

    # Adaptive: Sweet spot (42.5% reduction, <1% quality loss)
    results.append(
        {
            "config": "adaptive",
            "ttt_iterations": "adaptive",
            "num_params": 44122112,
            "train_losses": [4.76, 4.54, 4.43],
            "val_perplexities": [117.5, 99.8, 96.1],
            "test_loss": 4.56,
            "test_perplexity": 95.5,  # Only 0.9 higher than Fixed-4
            "flops_per_token": 1.98e10,  # 37% reduction vs Fixed-4
            "ttt_stats": {
                "avg_iterations": 1.98,
                "flops_reduction": 0.371,  # 37.1% reduction
                "allocation_distribution": {"1": 0.31, "2": 0.42, "4": 0.27},
            },
        }
    )

    # Save results
    os.makedirs("experiments/results", exist_ok=True)

    for result in results:
        config = result["config"]
        iters = result["ttt_iterations"]
        filename = f"demo_{config}_{iters}.json"
        filepath = os.path.join("experiments/results", filename)

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Created: {filepath}")

    # Print summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Test PPL':<12} {'vs Best':<10} {'FLOPs ↓':<10}")
    print("-" * 60)

    best_ppl = float(min(r["test_perplexity"] for r in results))  # type: ignore[arg-type]

    for result in results:
        config = result["config"]
        iters = result.get("ttt_iterations", "adaptive")
        config_str = f"{config}-{iters}"

        ppl = float(result["test_perplexity"])
        ppl_diff = ((ppl - best_ppl) / best_ppl) * 100

        if "ttt_stats" in result:
            stats = result["ttt_stats"]
            flops_red = stats["flops_reduction"]
            print(f"{config_str:<20} {ppl:<12.2f} +{ppl_diff:<9.1f}% {flops_red * 100:<10.1f}%")
        else:
            print(f"{config_str:<20} {ppl:<12.2f} +{ppl_diff:<9.1f}% {'-':<10}")

    print("\n" + "=" * 60)
    print("✅ Demo results generated!")
    print("\nNext: Run visualization")
    print("  uv run python experiments/analyze_wikitext2.py")
    print("=" * 60)


if __name__ == "__main__":
    generate_demo_results()
