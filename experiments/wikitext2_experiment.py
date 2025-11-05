"""
WikiText-2 Language Modeling Experiments (Days 4-5).

This script runs baseline and adaptive TTT experiments on WikiText-2.

Usage:
    # Run all experiments
    uv run python experiments/wikitext2_experiment.py --mode all

    # Run specific baseline
    uv run python experiments/wikitext2_experiment.py --mode baseline --ttt_iterations 2

    # Run adaptive experiment
    uv run python experiments/wikitext2_experiment.py --mode adaptive
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ponderttt.data import get_wikitext_dataloaders
from ponderttt.models import TransformerConfig, TransformerTTT


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_per_token(config: TransformerConfig, ttt_iterations: float) -> float:
    """
    Estimate FLOPs per token for the model.

    Args:
        config: Model configuration
        ttt_iterations: Average TTT iterations (can be fractional for adaptive)

    Returns:
        Estimated FLOPs per token
    """
    d = config.hidden_dim
    d_ffn = config.ffn_dim
    d_ttt = config.ttt_dim
    n_layers = config.num_layers

    # Standard transformer layer FLOPs (per token, per layer)
    # Self-attention: ~4 * d^2 (QKV proj + output proj)
    # FFN: ~8 * d * d_ffn
    attention_flops = 4 * d * d
    ffn_flops = 8 * d * d_ffn
    standard_layer_flops = attention_flops + ffn_flops

    # TTT layer FLOPs (replaces one attention layer)
    # Each TTT iteration: ~4 * d * d_ttt (forward + backward approximation)
    ttt_layer_flops = ttt_iterations * 4 * d * d_ttt + ffn_flops

    # Total FLOPs
    total_flops = (n_layers - 1) * standard_layer_flops + ttt_layer_flops

    return total_flops


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Evaluate model on dataloader.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to use
        max_batches: Maximum number of batches (for quick evaluation)

    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_ttt_stats = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels, return_stats=True)

            loss = outputs["loss"]
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

            # Collect TTT stats if available
            if "ttt_stats" in outputs:
                all_ttt_stats.extend(outputs["ttt_stats"])

            # Update progress bar
            current_loss = total_loss / total_tokens
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Aggregate TTT stats
    ttt_summary = {}
    if all_ttt_stats:
        # Extract efficiency metrics
        if "efficiency" in all_ttt_stats[0]:
            avg_iterations = float(
                np.mean([s["efficiency"]["avg_iterations"] for s in all_ttt_stats])
            )
            flops_reduction = float(
                np.mean([s["efficiency"]["flops_reduction"] for s in all_ttt_stats])
            )

            ttt_summary = {
                "avg_iterations": avg_iterations,
                "flops_reduction": flops_reduction,
            }

            # Get allocation distribution if available
            if "distribution" in all_ttt_stats[0]:
                distributions = [s["distribution"] for s in all_ttt_stats]
                # Average distributions
                avg_dist = {}
                for key in distributions[0].keys():
                    avg_dist[key] = float(np.mean([d[key] for d in distributions]))

                ttt_summary["allocation_distribution"] = avg_dist  # type: ignore[assignment,unsupported-operation]

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "ttt_stats": ttt_summary,
    }


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        max_batches: Maximum batches per epoch

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track loss
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_tokens


def run_baseline_experiment(
    ttt_iterations: int,
    num_epochs: int = 3,
    max_train_batches: Optional[int] = 500,
    max_eval_batches: Optional[int] = 100,
    device_str: str = "cuda",
) -> Dict:
    """
    Run baseline experiment with fixed TTT iterations.

    Args:
        ttt_iterations: Fixed number of TTT iterations
        num_epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch
        max_eval_batches: Maximum evaluation batches
        device: Device to use

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Running baseline: Fixed-{ttt_iterations}")
    print(f"{'=' * 60}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading WikiText-2...")
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        max_length=256,  # Shorter for faster experiments
        batch_size=8,
        num_workers=0,
    )

    # Create model
    print("Creating model...")
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        max_seq_len=256,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=3,
        ttt_dim=256,
        ttt_iterations=ttt_iterations,
        use_adaptive_ttt=False,
    )

    model = TransformerTTT(config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    train_losses = []
    val_perplexities = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, max_train_batches)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, max_eval_batches)
        val_perplexities.append(val_metrics["perplexity"])
        print(f"Val perplexity: {val_metrics['perplexity']:.2f}")

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device, max_eval_batches)

    # Estimate FLOPs
    flops_per_token = estimate_flops_per_token(config, ttt_iterations)

    results = {
        "config": "baseline",
        "ttt_iterations": ttt_iterations,
        "num_params": num_params,
        "train_losses": train_losses,
        "val_perplexities": val_perplexities,
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "flops_per_token": flops_per_token,
    }

    return results


def run_adaptive_experiment(
    num_epochs: int = 3,
    max_train_batches: Optional[int] = 500,
    max_eval_batches: Optional[int] = 100,
    device_str: str = "cuda",
) -> Dict:
    """
    Run adaptive TTT experiment.

    Args:
        num_epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch
        max_eval_batches: Maximum evaluation batches
        device: Device to use

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 60}")
    print("Running adaptive TTT with entropy metric")
    print(f"{'=' * 60}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading WikiText-2...")
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        dataset_name="wikitext-2-raw-v1",
        tokenizer_name="gpt2",
        max_length=256,
        batch_size=8,
        num_workers=0,
    )

    # Create model with adaptive TTT
    print("Creating model with adaptive TTT...")
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        max_seq_len=256,
        dropout=0.1,
        use_ttt=True,
        ttt_layer_idx=3,
        ttt_dim=256,
        ttt_iterations=2,  # Base iterations (for calibration)
        use_adaptive_ttt=True,
        ttt_difficulty_metric="entropy",
        ttt_buckets=[1, 2, 4],
        ttt_target_distribution=[0.3, 0.4, 0.3],
    )

    model = TransformerTTT(config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    train_losses = []
    val_perplexities = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, max_train_batches)
        train_losses.append(train_loss)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, max_eval_batches)
        val_perplexities.append(val_metrics["perplexity"])
        print(f"Val perplexity: {val_metrics['perplexity']:.2f}")

        # Print TTT stats
        if val_metrics["ttt_stats"]:
            stats = val_metrics["ttt_stats"]
            print(f"  Avg iterations: {stats.get('avg_iterations', 0):.2f}")
            print(f"  FLOPs reduction: {stats.get('flops_reduction', 0) * 100:.1f}%")

            if "allocation_distribution" in stats:
                dist = stats["allocation_distribution"]
                print(f"  Allocation: {dist}")

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, test_loader, device, max_eval_batches)

    # Estimate FLOPs (use average iterations from test)
    avg_iterations = test_metrics["ttt_stats"].get("avg_iterations", 2.0)
    flops_per_token = estimate_flops_per_token(config, avg_iterations)

    results = {
        "config": "adaptive",
        "num_params": num_params,
        "train_losses": train_losses,
        "val_perplexities": val_perplexities,
        "test_loss": test_metrics["loss"],
        "test_perplexity": test_metrics["perplexity"],
        "flops_per_token": flops_per_token,
        "ttt_stats": test_metrics["ttt_stats"],
    }

    return results


def save_results(results: Dict, output_dir: str = "experiments/results"):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    config_name = results.get("config", "unknown")
    ttt_iters = results.get("ttt_iterations", "adaptive")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"{config_name}_{ttt_iters}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    results = convert_types(results)  # type: ignore[assignment]

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 TTT Experiments")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "baseline", "adaptive"],
        help="Experiment mode",
    )
    parser.add_argument(
        "--ttt_iterations",
        type=int,
        default=2,
        help="TTT iterations for baseline mode",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--max_train_batches", type=int, default=500, help="Max training batches per epoch"
    )
    parser.add_argument("--max_eval_batches", type=int, default=100, help="Max evaluation batches")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use"
    )

    args = parser.parse_args()

    all_results = []

    if args.mode in ["all", "baseline"]:
        # Run baselines
        if args.mode == "all":
            iterations_to_test = [1, 2, 4]
        else:
            iterations_to_test = [args.ttt_iterations]

        for iters in iterations_to_test:
            results = run_baseline_experiment(
                ttt_iterations=iters,
                num_epochs=args.num_epochs,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                device_str=args.device,
            )
            save_results(results)
            all_results.append(results)

    if args.mode in ["all", "adaptive"]:
        # Run adaptive
        results = run_adaptive_experiment(
            num_epochs=args.num_epochs,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            device_str=args.device,
        )
        save_results(results)
        all_results.append(results)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for result in all_results:
        config = result["config"]
        iters = result.get("ttt_iterations", "adaptive")
        ppl = result["test_perplexity"]
        flops = result["flops_per_token"]

        print(f"{config}-{iters}:")
        print(f"  Test Perplexity: {ppl:.2f}")
        print(f"  FLOPs/token: {flops:.2e}")

        if "ttt_stats" in result and result["ttt_stats"]:
            stats = result["ttt_stats"]
            if "avg_iterations" in stats:
                print(f"  Avg iterations: {stats['avg_iterations']:.2f}")
            if "flops_reduction" in stats:
                print(f"  FLOPs reduction: {stats['flops_reduction'] * 100:.1f}%")

        print()


if __name__ == "__main__":
    main()
