"""
Full comparison suite for PonderTTT.

Compares 8 methods with fair parameter counts and consistent evaluation:
1-4. Uniform K=1, 2, 4, 8 (iterative, fixed steps)
5. Heuristic adaptive (entropy-based)
6-8. Learned adaptive (λ=0.01/0.05, with/without target)

All methods use the same base architecture (60.10M parameters via DummyPolicy).

Note: Official TTT (analytic baseline) is compared separately in convergence_analysis.py
due to different mini-batch processing requirements.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ponderttt.data.wikitext import get_wikitext2_dataloaders
from ponderttt.models import (
    IterativeTransformerConfig,
    IterativeTransformerTTT,
    EntropyBasedPolicy,
    UniformPolicy,
)
from ponderttt.utils.metrics import compute_perplexity


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ComparisonMethod:
    """
    Configuration for a single comparison method.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: IterativeTransformerConfig,
        policy_type: Optional[str] = None,
        fixed_steps: Optional[int] = None,
    ):
        self.name = name
        self.description = description
        self.config = config
        self.policy_type = policy_type
        self.fixed_steps = fixed_steps

    def create_model(self, device: str) -> IterativeTransformerTTT:
        """Create model instance."""
        model = IterativeTransformerTTT(self.config).to(device)

        # For heuristic policies, attach the policy after model creation
        if self.policy_type == 'entropy':
            for block in model.blocks:
                if hasattr(block, 'halting_policy') and block.halting_policy is not None:
                    # Replace with entropy-based policy
                    block.halting_policy = EntropyBasedPolicy(
                        lm_head=model.lm_head,
                        step_options=self.config.step_options,
                    ).to(device)
        elif self.policy_type == 'uniform':
            for block in model.blocks:
                if hasattr(block, 'halting_policy') and block.halting_policy is not None:
                    block.halting_policy = UniformPolicy(
                        fixed_steps=self.fixed_steps,
                        step_options=self.config.step_options,
                    ).to(device)

        return model


def get_all_methods() -> List[ComparisonMethod]:
    """
    Define all 8 comparison methods.

    Returns:
        List of ComparisonMethod instances
    """
    base_config = {
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'ffn_dim': 2048,
        'use_iterative_ttt': True,
        'ttt_layer_indices': [2, 3, 4],
        'fast_weight_hidden_dim': 64,
        'step_options': [1, 2, 4, 8],
        'policy_use_lstm': True,
        'policy_pooling': 'none',
    }

    methods = []

    # Method 1: Uniform K=1
    methods.append(ComparisonMethod(
        name='uniform_k1',
        description='Uniform K=1 (minimal compute)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=1,
            use_learned_policy=False,
            lambda_compute=0.0,
        ),
        policy_type='uniform',
        fixed_steps=1,
    ))

    # Method 2: Uniform K=2
    methods.append(ComparisonMethod(
        name='uniform_k2',
        description='Uniform K=2',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=2,
            use_learned_policy=False,
            lambda_compute=0.0,
        ),
        policy_type='uniform',
        fixed_steps=2,
    ))

    # Method 3: Uniform K=4
    methods.append(ComparisonMethod(
        name='uniform_k4',
        description='Uniform K=4',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=4,
            use_learned_policy=False,
            lambda_compute=0.0,
        ),
        policy_type='uniform',
        fixed_steps=4,
    ))

    # Method 4: Uniform K=8
    methods.append(ComparisonMethod(
        name='uniform_k8',
        description='Uniform K=8 (maximum compute)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=8,
            use_learned_policy=False,
            lambda_compute=0.0,
        ),
        policy_type='uniform',
        fixed_steps=8,
    ))

    # Method 5: Entropy-based heuristic
    methods.append(ComparisonMethod(
        name='heuristic_entropy',
        description='Heuristic (entropy-based difficulty)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=8,
            use_learned_policy=True,  # Will be replaced with heuristic
            lambda_compute=0.0,
        ),
        policy_type='entropy',
    ))

    # Method 6: Learned policy (λ=0.01, target=4)
    methods.append(ComparisonMethod(
        name='learned_lambda001_target4',
        description='Learned policy (λ=0.01, target avg=4)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=8,
            use_learned_policy=True,
            lambda_compute=0.01,
            target_avg_steps=4.0,
        ),
        policy_type=None,  # Real learned policy
    ))

    # Method 7: Learned policy (λ=0.05, target=4)
    methods.append(ComparisonMethod(
        name='learned_lambda005_target4',
        description='Learned policy (λ=0.05, target avg=4)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=8,
            use_learned_policy=True,
            lambda_compute=0.05,
            target_avg_steps=4.0,
        ),
        policy_type=None,
    ))

    # Method 8: Learned policy (λ=0.01, no target)
    methods.append(ComparisonMethod(
        name='learned_lambda001_notarget',
        description='Learned policy (λ=0.01, no target)',
        config=IterativeTransformerConfig(
            **base_config,
            max_steps=8,
            use_learned_policy=True,
            lambda_compute=0.01,
            target_avg_steps=None,
        ),
        policy_type=None,
    ))

    return methods


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_steps: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            num_steps=num_steps,
            return_stats=False,
        )

        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_steps: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
) -> Dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_stats = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                num_steps=num_steps,
                return_stats=True,
            )

            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1

            if 'stats' in outputs and outputs['stats']:
                all_stats.extend(outputs['stats'])

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = compute_perplexity(avg_loss)

    # Aggregate stats
    avg_steps = None
    if all_stats:
        step_values = [s.get('avg_steps') for s in all_stats if 'avg_steps' in s]
        if step_values:
            avg_steps = sum(step_values) / len(step_values)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'avg_steps': avg_steps,
    }


def run_single_method(
    method: ComparisonMethod,
    seed: int,
    num_epochs: int,
    max_train_batches: Optional[int],
    max_eval_batches: Optional[int],
    device: str,
    results_dir: Path,
) -> Dict:
    """
    Run experiment for a single method with a single seed.

    Args:
        method: ComparisonMethod instance
        seed: Random seed
        num_epochs: Number of training epochs
        max_train_batches: Max batches per epoch (None = full)
        max_eval_batches: Max eval batches (None = full)
        device: Device to use
        results_dir: Directory to save results

    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Method: {method.name} ({method.description})")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    train_loader, val_loader, test_loader = get_wikitext2_dataloaders(
        batch_size=8,
        max_length=256,
    )

    # Create model
    model = method.create_model(device)
    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Training
    best_val_ppl = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_perplexity': [],
    }

    num_steps = method.fixed_steps if method.fixed_steps else None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            num_steps=num_steps,
            max_batches=max_train_batches,
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, device,
            num_steps=num_steps,
            max_batches=max_eval_batches,
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_perplexity'].append(val_metrics['perplexity'])

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val PPL: {val_metrics['perplexity']:.2f}")
        if val_metrics['avg_steps']:
            print(f"  Avg steps: {val_metrics['avg_steps']:.2f}")

        if val_metrics['perplexity'] < best_val_ppl:
            best_val_ppl = val_metrics['perplexity']

    # Final test evaluation
    print("\nFinal test evaluation...")
    test_metrics = evaluate(
        model, test_loader, device,
        num_steps=num_steps,
        max_batches=max_eval_batches,
    )

    print(f"  Test PPL: {test_metrics['perplexity']:.2f}")
    if test_metrics['avg_steps']:
        print(f"  Avg steps: {test_metrics['avg_steps']:.2f}")

    # Collect results
    results = {
        'method': method.name,
        'description': method.description,
        'seed': seed,
        'num_params': num_params,
        'best_val_perplexity': best_val_ppl,
        'test_perplexity': test_metrics['perplexity'],
        'test_loss': test_metrics['loss'],
        'test_avg_steps': test_metrics['avg_steps'],
        'history': history,
        'config': {
            'num_epochs': num_epochs,
            'max_train_batches': max_train_batches,
            'max_eval_batches': max_eval_batches,
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{method.name}_seed{seed}_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Full comparison suite')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Specific methods to run (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--max_train_batches', type=int, default=None,
                       help='Max training batches per epoch')
    parser.add_argument('--max_eval_batches', type=int, default=None,
                       help='Max evaluation batches')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--results_dir', type=str, default='experiments/results/comparison',
                       help='Results directory')

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get all methods
    all_methods = get_all_methods()

    # Filter methods if specified
    if args.methods:
        all_methods = [m for m in all_methods if m.name in args.methods]

    print(f"\nRunning {len(all_methods)} methods with {len(args.seeds)} seeds each")
    print(f"Total experiments: {len(all_methods) * len(args.seeds)}")

    # Run all experiments
    all_results = []

    for method in all_methods:
        for seed in args.seeds:
            result = run_single_method(
                method=method,
                seed=seed,
                num_epochs=args.num_epochs,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                device=args.device,
                results_dir=results_dir,
            )
            all_results.append(result)

    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved in: {results_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
