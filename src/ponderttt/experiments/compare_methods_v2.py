"""
Compare gating methods for adaptive Test-Time Training (v2 - Refactored).

This is the refactored version using modular gating strategies and eval_core.

Available Methods:
    - SKIP: Always skip TTT updates (baseline)
    - UPDATE: Always apply TTT updates
    - Random: Update with probability p
    - Oracle: Upper bound using ground-truth advantage
    - Threshold: Fixed threshold on reconstruction loss
    - EMA: Adaptive threshold targeting update rate
    - Reconstruction: PonderTTT method (reconstruction loss gating)

Usage:
    python -m ponderttt.experiments.compare_methods_v2 --model_scale 125m --method oracle
    python -m ponderttt.experiments.compare_methods_v2 --model_scale 350m --method ema
"""

import argparse
from pathlib import Path
from typing import Optional, Literal

import jax
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTModel, TTTTransformerLM
from ..gating import (
    GatingStrategy,
    FixedGating,
    RandomGating,
    ThresholdGating,
    EMAGating,
    ReconstructionGating,
)
from .eval_core import (
    ChunkResult,
    EvalResult,
    evaluate_oracle,
    evaluate_with_gating,
    print_eval_summary,
)
from .jit_helpers import compute_both_losses, get_ttt_loss_from_stats


# Model scale to HuggingFace model name mapping
MODEL_SCALE_MAP = {
    "125m": "gpt2",
    "350m": "gpt2-medium",
    "1b": "gpt2-large",
    "xl": "gpt2-xl",
}


def create_gating_strategy(
    method: str,
    update_rate: float = 0.5,
    threshold: float = 0.5,
    ema_alpha: float = 0.1,
) -> GatingStrategy:
    """Create a gating strategy by name.

    Args:
        method: One of "skip", "update", "random", "threshold", "ema", "reconstruction"
        update_rate: Target update rate for random/ema methods
        threshold: Fixed threshold for threshold method
        ema_alpha: EMA smoothing factor

    Returns:
        Configured GatingStrategy instance
    """
    method = method.lower()

    if method == "skip":
        return FixedGating(action="SKIP")
    elif method == "update":
        return FixedGating(action="UPDATE")
    elif method == "random":
        return RandomGating(probability=update_rate)
    elif method == "threshold":
        return ThresholdGating(threshold=threshold, signal="ttt_loss_init")
    elif method == "ema":
        return EMAGating(
            target_update_rate=update_rate,
            ema_alpha=ema_alpha,
            signal="ttt_loss_init",
        )
    elif method == "reconstruction":
        return ReconstructionGating(
            mode="adaptive",
            target_rate=update_rate,
            ema_alpha=ema_alpha,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Available: skip, update, random, threshold, ema, reconstruction"
        )


def run_comparison(
    model_scale: str,
    methods: list[str],
    num_batches: int = 10,
    batch_size: int = 1,
    update_rate: float = 0.5,
    language: str = "Python",
    split: str = "test",
    skip_examples: int = 0,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> dict[str, EvalResult]:
    """Run comparison of gating methods.

    Args:
        model_scale: One of "125m", "350m", "1b", "xl"
        methods: List of method names to compare
        num_batches: Number of batches per method
        batch_size: Batch size
        update_rate: Target update rate for adaptive methods
        language: Programming language for dataset
        split: Dataset split
        skip_examples: Examples to skip
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dictionary mapping method name to EvalResult
    """
    # Load model once
    model_name = MODEL_SCALE_MAP[model_scale]
    tokenizer = get_tokenizer(model_name)

    print(f"\nLoading {model_name}...")
    model, config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )

    results: dict[str, EvalResult] = {}

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {method.upper()}")
        print(f"{'=' * 60}")

        # Create data iterator (fresh for each method)
        data_iter = create_data_iterator(
            tokenizer=tokenizer,
            split=split,
            language=language,
            batch_size=batch_size,
            seq_length=1024,
            chunk_size=512,
            max_examples=batch_size * num_batches * 2,
            skip_examples=skip_examples,
            num_workers=32,
        )

        if method.lower() == "oracle":
            # Oracle uses special evaluation
            result = evaluate_oracle(
                model=model,
                data_iter=data_iter,
                num_batches=num_batches,
                chunk_size=512,
            )
        else:
            # Create gating strategy
            gating = create_gating_strategy(
                method=method,
                update_rate=update_rate,
            )

            result = evaluate_with_gating(
                model=model,
                data_iter=data_iter,
                gating=gating,
                num_batches=num_batches,
                chunk_size=512,
                compute_oracle=True,
            )

        result.method_name = method
        results[method] = result

        # Print summary
        print_eval_summary(result)

    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary DataFrame
        summaries = [r.summary() for r in results.values()]
        df = pd.DataFrame(summaries)

        csv_path = output_dir / f"comparison_{model_scale}_{language}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare gating methods for TTT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_scale",
        type=str,
        default="125m",
        choices=["125m", "350m", "1b", "xl"],
        help="Model scale",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["skip", "update", "oracle", "reconstruction"],
        help="Methods to compare",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--update_rate",
        type=float,
        default=0.5,
        help="Target update rate for adaptive methods",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Python",
        help="Programming language",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/comparison",
        help="Output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results = run_comparison(
        model_scale=args.model_scale,
        methods=args.methods,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        update_rate=args.update_rate,
        language=args.language,
        split=args.split,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    for method, result in results.items():
        summary = result.summary()
        print(f"\n{method.upper()}:")
        print(f"  Loss: {summary['avg_loss']:.4f}")
        print(f"  Update Rate: {summary['update_rate']:.1%}")
        print(f"  Oracle Accuracy: {summary['oracle_accuracy']:.1%}")


if __name__ == "__main__":
    main()
