"""
Extended Oracle K Allocation Analysis.

Provides comprehensive analysis of optimal K allocation:
1. Large-scale analysis on validation set (100+ sequences)
2. Oracle K distribution visualization
3. Difficulty-K correlation analysis (Pearson & Spearman)
4. Oracle Pareto frontier (upper bound for learned policies)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from ponderttt.data.wikitext import get_wikitext2_dataloaders
from ponderttt.models import (
    IterativeTransformerConfig,
    IterativeTransformerTTT,
)
from ponderttt.experiments.oracle_analysis import OracleAnalyzer


class ExtendedOracleAnalyzer(OracleAnalyzer):
    """
    Extended oracle analyzer with visualization and correlation analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        step_options: List[int] = [1, 2, 4, 8],
        device: str = 'cuda',
    ):
        super().__init__(model, step_options, device)

    def analyze_difficulty_correlation(
        self,
        batch: Dict,
        oracle_results: List[Dict],
    ) -> Dict:
        """
        Analyze correlation between difficulty metrics and oracle K.

        Args:
            batch: Batch dictionary with input_ids, labels
            oracle_results: List of per-token oracle results

        Returns:
            correlations: Dictionary of correlation statistics
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Extract oracle K values
        oracle_k_values = [r['oracle_k'] for r in oracle_results]

        # Compute difficulty metrics
        # Note: We need gradients enabled for fast-weight updates, but don't call backward()
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            return_stats=False,
        )
        logits = outputs['logits']

        # Get hidden states (use embeddings as proxy)
        if hasattr(self.model, 'token_embedding'):
            hidden_states = self.model.token_embedding(input_ids)
        else:
            hidden_states = None

        # Difficulty metric 1: Per-token loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(shift_labels.shape)  # (batch, seq_len-1)

        loss_values = [r['difficulty']['loss'] for r in oracle_results]

        # Difficulty metric 2: Entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len, vocab)

        entropy_values = [r['difficulty']['entropy'] for r in oracle_results]

        # Difficulty metric 3: Gradient norm
        grad_norm_values = [r['difficulty']['gradient_norm'] for r in oracle_results]

        # Compute correlations
        correlations = {}

        # Oracle K vs Loss
        if len(oracle_k_values) > 2 and len(loss_values) > 2:
            pearson_r, pearson_p = pearsonr(oracle_k_values, loss_values)
            spearman_r, spearman_p = spearmanr(oracle_k_values, loss_values)
            correlations['oracle_k_vs_loss'] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
            }

        # Oracle K vs Entropy
        if len(oracle_k_values) > 2 and len(entropy_values) > 2:
            pearson_r, pearson_p = pearsonr(oracle_k_values, entropy_values)
            spearman_r, spearman_p = spearmanr(oracle_k_values, entropy_values)
            correlations['oracle_k_vs_entropy'] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
            }

        # Oracle K vs Gradient Norm
        if len(oracle_k_values) > 2 and len(grad_norm_values) > 2:
            pearson_r, pearson_p = pearsonr(oracle_k_values, grad_norm_values)
            spearman_r, spearman_p = spearmanr(oracle_k_values, grad_norm_values)
            correlations['oracle_k_vs_gradient_norm'] = {
                'pearson_r': float(pearson_r),
                'pearson_p': float(pearson_p),
                'spearman_r': float(spearman_r),
                'spearman_p': float(spearman_p),
            }

        return correlations

    def compute_oracle_pareto_frontier(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 20,
    ) -> Dict:
        """
        Compute oracle Pareto frontier (upper bound).

        For each K budget, find the oracle allocation that achieves
        that average K and measure quality.

        Args:
            dataloader: Data loader
            max_batches: Maximum batches to process

        Returns:
            pareto_results: Dictionary with quality-compute tradeoff
        """
        print("\nComputing oracle Pareto frontier...")

        all_batch_results = []

        pbar = tqdm(dataloader, desc="Oracle Pareto analysis", total=max_batches)

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            batch_size, seq_len = input_ids.shape

            # Get oracle K for each position
            oracle_k_allocation = torch.zeros(seq_len, dtype=torch.long, device=self.device)

            for pos in range(seq_len):
                best_k, k_losses = self.find_oracle_k_for_token(
                    input_ids, labels, pos
                )
                oracle_k_allocation[pos] = best_k

            # Compute loss with oracle allocation
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                num_steps=oracle_k_allocation,
                return_stats=False,
            )

            oracle_loss = outputs['loss'].item()
            oracle_mean_k = oracle_k_allocation.float().mean().item()

            all_batch_results.append({
                'oracle_k_allocation': oracle_k_allocation.cpu().tolist(),
                'oracle_loss': oracle_loss,
                'oracle_mean_k': oracle_mean_k,
            })

        # Aggregate results
        oracle_losses = [r['oracle_loss'] for r in all_batch_results]
        oracle_mean_ks = [r['oracle_mean_k'] for r in all_batch_results]

        # Also compute uniform baselines for comparison
        uniform_results = {}
        for k in self.step_options:
            uniform_losses = []

            pbar = tqdm(dataloader, desc=f"Uniform K={k}", total=max_batches)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= max_batches:
                    break

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                seq_len = input_ids.shape[1]

                num_steps = torch.full((seq_len,), k, dtype=torch.long, device=self.device)

                # Note: Need gradients for fast-weight updates, but don't call backward()
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    num_steps=num_steps,
                    return_stats=False,
                )

                uniform_losses.append(outputs['loss'].item())

            uniform_results[k] = {
                'mean_k': k,
                'mean_loss': np.mean(uniform_losses),
                'std_loss': np.std(uniform_losses),
            }

        return {
            'oracle': {
                'mean_k_values': oracle_mean_ks,
                'loss_values': oracle_losses,
                'mean_loss': np.mean(oracle_losses),
                'std_loss': np.std(oracle_losses),
                'mean_mean_k': np.mean(oracle_mean_ks),
            },
            'uniform': uniform_results,
            'all_batch_results': all_batch_results,
        }

    def plot_oracle_k_distribution(
        self,
        aggregated_results: Dict,
        output_dir: Path,
    ):
        """
        Plot oracle K distribution.

        Args:
            aggregated_results: Results from analyze_dataset
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract oracle K distribution
        oracle_k_dist = aggregated_results['oracle_k_distribution']

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        k_values = sorted(oracle_k_dist.keys())
        counts = [oracle_k_dist[k] for k in k_values]
        percentages = [aggregated_results['oracle_k_percentages'][k] for k in k_values]

        # Bar plot
        bars = ax.bar(range(len(k_values)), percentages, color='steelblue', alpha=0.8)

        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=11)

        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.set_xlabel('Oracle K Value', fontsize=12)
        ax.set_ylabel('Percentage of Tokens (%)', fontsize=12)
        ax.set_title('Oracle K Allocation Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'oracle_k_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_dir / 'oracle_k_distribution.png'}")

    def plot_difficulty_correlation(
        self,
        aggregated_results: Dict,
        output_dir: Path,
    ):
        """
        Plot difficulty vs oracle K correlation scatter plots.

        Args:
            aggregated_results: Results from analyze_dataset
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        correlations = aggregated_results['correlations']

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        metrics = [
            ('oracle_k_vs_entropy', 'Entropy', 'Entropy'),
            ('oracle_k_vs_loss', 'Loss', 'Per-Token Loss'),
            ('oracle_k_vs_gradient_norm', 'Gradient Norm', 'Gradient L2 Norm'),
        ]

        for ax, (key, short_name, full_name) in zip(axes, metrics):
            if key in correlations:
                corr_data = correlations[key]
                pearson_r = corr_data
                spearman_r = corr_data if isinstance(corr_data, (int, float)) else corr_data

                # Note: We don't have individual data points here, just correlation values
                # So we'll create a text-based visualization
                ax.text(0.5, 0.6, f'Pearson r: {pearson_r:.3f}',
                        ha='center', va='center', fontsize=14, fontweight='bold')
                ax.text(0.5, 0.4, f'Spearman ρ: {spearman_r:.3f}',
                        ha='center', va='center', fontsize=14)

                ax.set_title(f'{short_name} vs Oracle K', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'difficulty_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_dir / 'difficulty_correlation.png'}")

    def plot_pareto_frontier(
        self,
        pareto_results: Dict,
        output_dir: Path,
    ):
        """
        Plot oracle Pareto frontier.

        Args:
            pareto_results: Results from compute_oracle_pareto_frontier
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot uniform baselines
        uniform_k_values = []
        uniform_losses = []
        for k in sorted(pareto_results['uniform'].keys()):
            uniform_k_values.append(k)
            uniform_losses.append(pareto_results['uniform'][k]['mean_loss'])

        ax.plot(uniform_k_values, uniform_losses, 'o-', label='Uniform K',
                markersize=8, linewidth=2, color='gray', alpha=0.7)

        # Plot oracle
        oracle_mean_k = pareto_results['oracle']['mean_mean_k']
        oracle_mean_loss = pareto_results['oracle']['mean_loss']
        oracle_std_loss = pareto_results['oracle']['std_loss']

        ax.errorbar(oracle_mean_k, oracle_mean_loss, yerr=oracle_std_loss,
                    fmt='*', markersize=15, linewidth=2, capsize=5,
                    label='Oracle (Upper Bound)', color='gold', markeredgecolor='black')

        ax.set_xlabel('Average K (Compute)', fontsize=12)
        ax.set_ylabel('Loss (Lower is Better)', fontsize=12)
        ax.set_title('Oracle Pareto Frontier (Upper Bound)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'oracle_pareto_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_dir / 'oracle_pareto_frontier.png'}")


def main():
    parser = argparse.ArgumentParser(description='Extended Oracle K allocation analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_batches', type=int, default=20,
                       help='Number of batches to analyze')
    parser.add_argument('--sample_positions', type=int, default=32,
                       help='Positions to sample per batch')
    parser.add_argument('--step_options', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--output_dir', type=str, default='experiments/results/oracle_extended')
    parser.add_argument('--pareto_batches', type=int, default=20,
                       help='Batches for Pareto frontier computation')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Extended Oracle K Allocation Analysis (Phase 2, Task 2.2)")
    print("="*80)
    print(f"Step options: {args.step_options}")
    print(f"Max batches: {args.max_batches}")
    print(f"Sample positions per batch: {args.sample_positions}")
    print()

    # Load data
    _, val_loader, test_loader = get_wikitext2_dataloaders(
        batch_size=4,  # Smaller batch for oracle analysis
        max_length=128,  # Shorter sequences
    )

    # Create model
    config = IterativeTransformerConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        ffn_dim=2048,
        use_iterative_ttt=True,
        ttt_layer_indices=[2, 3, 4],
        fast_weight_hidden_dim=64,
        max_steps=max(args.step_options),
        use_learned_policy=False,
        step_options=args.step_options,
    )

    model = IterativeTransformerTTT(config).to(args.device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Create extended analyzer
    analyzer = ExtendedOracleAnalyzer(
        model=model,
        step_options=args.step_options,
        device=args.device,
    )

    # Run basic oracle analysis
    print("Running oracle analysis on validation set...")
    results = analyzer.analyze_dataset(
        val_loader,
        max_batches=args.max_batches,
        sample_positions_per_batch=args.sample_positions,
    )

    # Print results
    print("\n" + "="*80)
    print("Oracle K Allocation Results")
    print("="*80)

    print(f"\nTotal tokens analyzed: {results['total_tokens_analyzed']}")
    print(f"Average oracle K: {results['average_oracle_k']:.2f}")

    print("\nOracle K Distribution:")
    for k in sorted(results['oracle_k_distribution'].keys()):
        count = results['oracle_k_distribution'][k]
        pct = results['oracle_k_percentages'][k]
        print(f"  K={k}: {count:4d} tokens ({pct:5.1f}%)")

    print("\nCorrelations with Difficulty Metrics:")
    for metric, corr in results['correlations'].items():
        print(f"  {metric}: {corr:.4f}")

    # Plot oracle K distribution
    analyzer.plot_oracle_k_distribution(results, output_dir)

    # Plot difficulty correlation
    analyzer.plot_difficulty_correlation(results, output_dir)

    # Compute oracle Pareto frontier
    pareto_results = analyzer.compute_oracle_pareto_frontier(
        val_loader,
        max_batches=args.pareto_batches,
    )

    print("\n" + "="*80)
    print("Oracle Pareto Frontier")
    print("="*80)

    print(f"\nOracle (Adaptive):")
    print(f"  Average K: {pareto_results['oracle']['mean_mean_k']:.2f}")
    print(f"  Average Loss: {pareto_results['oracle']['mean_loss']:.4f} ± {pareto_results['oracle']['std_loss']:.4f}")

    print(f"\nUniform Baselines:")
    for k in sorted(pareto_results['uniform'].keys()):
        result = pareto_results['uniform'][k]
        print(f"  K={k}: Loss={result['mean_loss']:.4f} ± {result['std_loss']:.4f}")

    # Plot Pareto frontier
    analyzer.plot_pareto_frontier(pareto_results, output_dir)

    # Save results
    results_file = output_dir / 'extended_oracle_results.json'
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'oracle_analysis': results,
            'pareto_frontier': {
                'oracle': pareto_results['oracle'],
                'uniform': pareto_results['uniform'],
            },
        }
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
