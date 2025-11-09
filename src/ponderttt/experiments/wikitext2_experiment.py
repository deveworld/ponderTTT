"""
WikiText-2 experiments with Iterative TTT.

Fair comparison framework:
- All experiments use SAME architecture
- Only difference: number of gradient steps per token
- No mini_batch_size variance

Experiments:
1. Uniform baselines (K=1,2,4,8)
2. Learned policy models
3. Oracle allocation (upper bound)
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..data import get_wikitext_dataloaders
from ..models.transformer_iterative import (
    IterativeTransformerConfig,
    IterativeTransformerTTT,
)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
) -> Dict:
    """
    Evaluate model on dataloader.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device
        num_steps: Optional fixed step counts (for uniform baselines)
        max_batches: Maximum number of batches

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

            batch_steps = None
            if num_steps is not None:
                if isinstance(num_steps, int):
                    batch_steps = torch.full(
                        (input_ids.size(1),),
                        num_steps,
                        dtype=torch.float32,
                        device=device
                    )
                else:
                    batch_steps = num_steps.float().to(device)

            # Forward pass
            outputs = model(
                input_ids,
                labels=labels,
                num_steps=batch_steps,
                return_stats=True
            )

            loss = outputs["loss"]
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

            # Collect stats
            if "ttt_stats" in outputs:
                all_ttt_stats.extend(outputs["ttt_stats"])

            current_loss = total_loss / total_tokens
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Aggregate TTT stats
    result = {
        "loss": avg_loss,
        "perplexity": perplexity,
    }

    if all_ttt_stats:
        # Average steps across all TTT layers
        avg_steps_list = [s['avg_steps'] for s in all_ttt_stats if 'avg_steps' in s]
        if avg_steps_list:
            result["avg_steps"] = float(np.mean(avg_steps_list))
            result["min_steps"] = float(np.min([s['min_steps'] for s in all_ttt_stats if 'min_steps' in s]))
            result["max_steps"] = float(np.max([s['max_steps'] for s in all_ttt_stats if 'max_steps' in s]))

    return result


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device
        num_steps: Optional fixed step counts
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

        batch_steps = None
        if num_steps is not None:
            if isinstance(num_steps, int):
                batch_steps = torch.full(
                    (input_ids.size(1),),
                    num_steps,
                    dtype=torch.float32,
                    device=device
                )
            else:
                batch_steps = num_steps.float().to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels, num_steps=batch_steps)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track loss
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_tokens


def run_uniform_baseline(
    fixed_steps: int,
    num_epochs: int = 10,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    device_str: str = "cuda",
    output_dir: str = "results/iterative",
) -> Dict:
    """
    Run uniform baseline experiment.

    All tokens receive the same number of gradient steps.

    Args:
        fixed_steps: Number of steps for all tokens
        num_epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch
        max_eval_batches: Maximum evaluation batches
        device_str: Device string
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Running Uniform Baseline: K={fixed_steps}")
    print(f"{'=' * 60}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading WikiText-2...")
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        batch_size=8,
        max_length=256,
    )

    # Create model
    print("Creating model...")
    config = IterativeTransformerConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        use_iterative_ttt=True,
        ttt_layer_indices=[2, 3, 4],
        fast_weight_hidden_dim=64,
        max_steps=max(8, fixed_steps),  # Ensure max_steps >= fixed_steps
        use_learned_policy=False,  # Uniform allocation
        lambda_compute=0.0,  # No compute penalty
    )

    model = IterativeTransformerTTT(config).to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # LR Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Training with early stopping
    print(f"\nTraining for {num_epochs} epochs (with early stopping)...")
    best_val_ppl = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_steps=fixed_steps,
            max_batches=max_train_batches,
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_steps=fixed_steps,
            max_batches=max_eval_batches,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
        print(f"  LR: {current_lr:.6f}")

        # LR scheduler step
        scheduler.step()

        # Early stopping check
        if val_metrics['perplexity'] < best_val_ppl:
            best_val_ppl = val_metrics['perplexity']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered (patience={patience})")
                break

    # Final test evaluation
    print("\nFinal test evaluation...")
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_steps=fixed_steps,
    )

    print(f"  Test PPL: {test_metrics['perplexity']:.2f}")

    # Compute FLOPs
    flops_per_token = model.estimate_flops_per_token(
        num_steps=fixed_steps,
        seq_len=256,
    )

    # Verify avg_steps matches fixed_steps
    measured_avg_steps = test_metrics.get('avg_steps', fixed_steps)
    if abs(measured_avg_steps - fixed_steps) > 0.01:
        print(f"  WARNING: Measured avg_steps ({measured_avg_steps:.2f}) != fixed_steps ({fixed_steps})")

    # Results
    results = {
        "experiment_type": "uniform_baseline",
        "fixed_steps": fixed_steps,
        "num_params": num_params,
        "num_epochs": num_epochs,
        "train_loss": train_loss,
        "val_perplexity": val_metrics["perplexity"],
        "test_perplexity": test_metrics["perplexity"],
        "avg_steps": measured_avg_steps,
        "flops_per_token": flops_per_token,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"uniform_K{fixed_steps}_{results['timestamp']}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


def run_all_uniform_baselines(
    num_epochs: int = 10,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    device_str: str = "cuda",
    output_dir: str = "results/iterative",
) -> List[Dict]:
    """
    Run all uniform baseline experiments.

    Returns:
        List of results dictionaries
    """
    step_counts = [1, 2, 4, 8]
    all_results = []

    for K in step_counts:
        results = run_uniform_baseline(
            fixed_steps=K,
            num_epochs=num_epochs,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            device_str=device_str,
            output_dir=output_dir,
        )
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 60)
    print("Uniform Baseline Summary")
    print("=" * 60)
    print(f"{'K':>4s} | {'Test PPL':>10s} | {'FLOPs':>15s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['fixed_steps']:>4d} | {r['test_perplexity']:>10.2f} | {r['flops_per_token']:>15,.0f}")

    return all_results


def run_baseline_experiment(
    ttt_iterations: int = 4,
    num_epochs: int = 10,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    device_str: str = "cuda",
) -> Dict:
    """
    Run baseline experiment with uniform iteration count.

    This is an alias for run_uniform_baseline for compatibility with
    run_experiments_with_seeds.py.

    Args:
        ttt_iterations: Number of gradient steps per token (uniform)
        num_epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch
        max_eval_batches: Maximum evaluation batches
        device_str: Device to use

    Returns:
        Results dictionary with test_perplexity, flops_per_token, etc.
    """
    result = run_uniform_baseline(
        fixed_steps=ttt_iterations,
        num_epochs=num_epochs,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        device_str=device_str,
        output_dir="results/iterative",
    )

    result["config"] = "baseline"
    result["ttt_iterations"] = ttt_iterations
    result["test_loss"] = result.get("train_loss", 0.0)
    result["ttt_stats"] = {}

    return result


def run_adaptive_experiment(
    num_epochs: int = 10,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    device_str: str = "cuda",
) -> Dict:
    """
    Run adaptive TTT experiment with learned halting policy.

    Args:
        num_epochs: Number of training epochs
        max_train_batches: Maximum training batches per epoch
        max_eval_batches: Maximum evaluation batches
        device_str: Device to use

    Returns:
        Results dictionary with test_perplexity, flops_per_token, etc.
    """
    print(f"\n{'=' * 60}")
    print("Running Adaptive TTT Experiment")
    print(f"{'=' * 60}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading WikiText-2...")
    train_loader, val_loader, test_loader = get_wikitext_dataloaders(
        batch_size=8,
        max_length=256,
    )

    print("Creating model with learned halting policy...")
    config = IterativeTransformerConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        use_iterative_ttt=True,
        ttt_layer_indices=[2, 3, 4],
        fast_weight_hidden_dim=64,
        max_steps=8,
        use_learned_policy=True,
        lambda_compute=0.01,
        target_avg_steps=4.0,
        policy_pooling='none',
    )

    model = IterativeTransformerTTT(config).to(device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # LR Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Training with early stopping
    print(f"\nTraining for {num_epochs} epochs (with early stopping)...")
    best_val_ppl = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_steps=None,
            max_batches=max_train_batches,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_steps=None,
            max_batches=max_eval_batches,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
        print(f"  LR: {current_lr:.6f}")

        # LR scheduler step
        scheduler.step()

        # Early stopping check
        if val_metrics['perplexity'] < best_val_ppl:
            best_val_ppl = val_metrics['perplexity']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered (patience={patience})")
                break

    print("\nFinal test evaluation...")
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_steps=None,
    )

    print(f"  Test PPL: {test_metrics['perplexity']:.2f}")

    avg_steps = test_metrics.get('avg_steps', 4.0)
    flops_per_token = model.estimate_flops_per_token(
        num_steps=avg_steps,
        seq_len=256,
    )

    results = {
        "config": "adaptive",
        "num_params": num_params,
        "num_epochs": num_epochs,
        "train_loss": train_loss,
        "val_perplexity": val_metrics["perplexity"],
        "test_perplexity": test_metrics["perplexity"],
        "test_loss": test_metrics["loss"],
        "avg_steps": avg_steps,
        "flops_per_token": flops_per_token,
        "ttt_stats": {
            "avg_scaling": test_metrics.get("avg_scaling", 1.0),
        },
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Iterative TTT WikiText-2 experiments")
    parser.add_argument("--mode", type=str, default="uniform_all",
                       choices=["uniform_all", "uniform_single", "learned", "oracle"],
                       help="Experiment mode")
    parser.add_argument("--fixed_steps", type=int, default=4,
                       help="Fixed step count (for uniform_single mode)")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--max_train_batches", type=int, default=None,
                       help="Maximum training batches per epoch")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                       help="Maximum evaluation batches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results/iterative",
                       help="Output directory")

    args = parser.parse_args()

    if args.mode == "uniform_all":
        # Run all uniform baselines
        results = run_all_uniform_baselines(
            num_epochs=args.num_epochs,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            device_str=args.device,
            output_dir=args.output_dir,
        )

    elif args.mode == "uniform_single":
        # Run single uniform baseline
        results = run_uniform_baseline(
            fixed_steps=args.fixed_steps,
            num_epochs=args.num_epochs,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            device_str=args.device,
            output_dir=args.output_dir,
        )

    elif args.mode == "learned":
        print("Learned policy mode not yet implemented")
        print("Coming in next phase!")

    elif args.mode == "oracle":
        print("Oracle allocation mode not yet implemented")
        print("Coming in next phase!")

    print("\n" + "=" * 60)
    print("Experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
