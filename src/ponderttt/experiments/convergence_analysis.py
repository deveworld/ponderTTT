"""
Convergence Analysis: Iterative vs Analytic TTT.

Analyzes the convergence gap between K-step iterative gradient descent
and the analytic closed-form solution. Helps answer:
1. Is K=4 sufficient to approximate the analytic solution?
2. How does convergence gap vary with K?
3. Are there cases where iterative is better (non-stationary)?
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ponderttt.data.wikitext import get_wikitext2_dataloaders
from ponderttt.models import (
    IterativeTTTLayerV2,
    OfficialTTTLayer,
)


class ConvergenceAnalyzer:
    """
    Analyzes convergence gap between iterative and analytic TTT.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        device: str = 'cuda',
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device = device

        # Create both layer types
        self.iterative_layer = IterativeTTTLayerV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            fast_weight_type='linear',
            ttt_loss_projection=False,
            lr_schedule='position_dependent',
            max_steps=16,  # Allow up to K=16 for analysis
        ).to(device)

        self.analytic_layer = OfficialTTTLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mini_batch_size=16,
            base_lr=0.1,
            use_learnable_lr=False,
        ).to(device)

        # Share parameters (Q, K, V projections)
        self._share_parameters()

    def _share_parameters(self):
        """Copy parameters from iterative to analytic layer."""
        # Copy Q, K, V projections
        self.analytic_layer.q_proj.weight.data = self.iterative_layer.q_proj.weight.data.clone()
        self.analytic_layer.k_proj.weight.data = self.iterative_layer.k_proj.weight.data.clone()
        self.analytic_layer.v_proj.weight.data = self.iterative_layer.v_proj.weight.data.clone()

        # Copy output projection
        self.analytic_layer.out_proj.weight.data = self.iterative_layer.out_proj.weight.data.clone()

        # Copy gate if present
        if hasattr(self.analytic_layer, 'gate_proj'):
            self.analytic_layer.gate_proj.weight.data = self.iterative_layer.gate_proj.weight.data.clone()

    @torch.no_grad()
    def compute_gap(
        self,
        x: torch.Tensor,
        k_values: List[int] = [1, 2, 4, 8, 16],
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute convergence gap for different K values.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            k_values: List of K values to test

        Returns:
            gaps: Dictionary mapping K → metrics
        """
        self.iterative_layer.eval()
        self.analytic_layer.eval()

        # Get analytic solution (ground truth)
        analytic_output, _ = self.analytic_layer(x, return_stats=False)

        gaps = {}

        for k in k_values:
            # Get iterative solution with K steps
            iterative_output, _, _ = self.iterative_layer(
                x,
                num_steps=k,
                return_stats=False,
            )

            # Compute various distance metrics
            l2_distance = torch.norm(iterative_output - analytic_output, p=2).item()
            l2_relative = l2_distance / torch.norm(analytic_output, p=2).item()

            mse = torch.mean((iterative_output - analytic_output) ** 2).item()
            mae = torch.mean(torch.abs(iterative_output - analytic_output)).item()

            # Cosine similarity
            flat_iter = iterative_output.reshape(-1)
            flat_anal = analytic_output.reshape(-1)
            cosine_sim = torch.nn.functional.cosine_similarity(
                flat_iter.unsqueeze(0),
                flat_anal.unsqueeze(0)
            ).item()

            gaps[k] = {
                'l2_distance': l2_distance,
                'l2_relative': l2_relative,
                'mse': mse,
                'mae': mae,
                'cosine_similarity': cosine_sim,
            }

        return gaps

    def analyze_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        k_values: List[int] = [1, 2, 4, 8, 16],
        max_batches: int = 50,
    ) -> Dict:
        """
        Analyze convergence gap across dataset.

        Args:
            dataloader: Data loader
            k_values: K values to test
            max_batches: Maximum batches to process

        Returns:
            results: Aggregated statistics
        """
        all_gaps = {k: [] for k in k_values}

        pbar = tqdm(dataloader, desc="Analyzing convergence", total=max_batches)

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)

            # Compute gaps for this batch
            gaps = self.compute_gap(input_ids, k_values)

            # Collect
            for k in k_values:
                all_gaps[k].append(gaps[k])

            # Update progress
            if batch_idx % 10 == 0:
                k4_gap = gaps[4]['l2_relative']
                pbar.set_postfix({'K=4 gap': f'{k4_gap:.4f}'})

        # Aggregate statistics
        results = {}
        for k in k_values:
            metrics = {}
            for metric_name in ['l2_distance', 'l2_relative', 'mse', 'mae', 'cosine_similarity']:
                values = [g[metric_name] for g in all_gaps[k]]
                metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                }
            results[k] = metrics

        return results


def theoretical_analysis() -> Dict:
    """
    Theoretical convergence analysis for convex case.

    Returns:
        analysis: Theoretical bounds and properties
    """
    analysis = {
        'convex_case': {
            'description': 'For strongly convex TTT loss with condition number κ',
            'convergence_rate': 'Linear: ||θ_k - θ*|| ≤ (1 - 1/κ)^k ||θ_0 - θ*||',
            'k_required_for_epsilon': 'k ≈ κ log(1/ε) for ε-accuracy',
            'typical_k': {
                'K=1': 'Poor approximation (large gap)',
                'K=2': 'Moderate approximation (~50% gap)',
                'K=4': 'Good approximation (~10% gap expected)',
                'K=8': 'Excellent approximation (<1% gap expected)',
            }
        },
        'non_convex_case': {
            'description': 'For non-convex TTT loss (typical in practice)',
            'convergence': 'No global convergence guarantees',
            'local_behavior': 'May converge to local minima',
            'iterative_advantage': 'More frequent smaller updates → better tracking of non-stationary data',
        },
        'implementation_notes': {
            'official_ttt': 'Single analytic update per mini-batch (equivalent to K=∞ in convex case)',
            'iterative_ttt': 'Explicit K-step GD (allows adaptive K per token)',
            'tradeoff': 'Accuracy (higher K) vs Compute (lower K)',
        }
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Convergence analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--output_dir', type=str, default='experiments/results/convergence')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Convergence Analysis: Iterative vs Analytic TTT")
    print("="*80)

    # Theoretical analysis
    print("\n### Theoretical Analysis ###\n")
    theory = theoretical_analysis()
    print(json.dumps(theory, indent=2))

    # Save theoretical analysis
    with open(output_dir / 'theoretical_analysis.json', 'w') as f:
        json.dump(theory, f, indent=2)

    # Empirical analysis
    print("\n### Empirical Analysis ###\n")

    # Load data
    _, _, test_loader = get_wikitext2_dataloaders(
        batch_size=8,
        max_length=256,
    )

    # Create analyzer
    analyzer = ConvergenceAnalyzer(
        hidden_dim=512,
        num_heads=8,
        device=args.device,
    )

    # Run analysis
    results = analyzer.analyze_dataset(
        test_loader,
        k_values=args.k_values,
        max_batches=args.max_batches,
    )

    # Print results
    print("\nConvergence Gap Analysis:")
    print("-" * 80)
    print(f"{'K':<5} {'L2 Relative (mean±std)':<30} {'Cosine Sim':<15} {'MSE':<15}")
    print("-" * 80)

    for k in args.k_values:
        l2_mean = results[k]['l2_relative']['mean']
        l2_std = results[k]['l2_relative']['std']
        cos_mean = results[k]['cosine_similarity']['mean']
        mse_mean = results[k]['mse']['mean']

        print(f"{k:<5} {l2_mean:.4f} ± {l2_std:.4f}            "
              f"{cos_mean:.4f}          {mse_mean:.6f}")

    print("-" * 80)

    # Interpretation
    print("\nInterpretation:")
    k4_gap = results[4]['l2_relative']['mean']
    if k4_gap < 0.05:
        print(f"✓ K=4 provides excellent approximation (gap={k4_gap:.4f} < 5%)")
    elif k4_gap < 0.15:
        print(f"○ K=4 provides good approximation (gap={k4_gap:.4f} < 15%)")
    else:
        print(f"✗ K=4 has significant gap (gap={k4_gap:.4f} ≥ 15%), consider higher K")

    # Save results
    results_file = output_dir / 'empirical_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, metrics in results.items():
            json_results[str(k)] = {
                metric_name: {
                    stat: float(value)
                    for stat, value in metric_stats.items()
                }
                for metric_name, metric_stats in metrics.items()
            }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Theoretical analysis saved to: {output_dir / 'theoretical_analysis.json'}")


if __name__ == '__main__':
    main()
