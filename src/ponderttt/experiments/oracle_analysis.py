"""
Oracle K Allocation Analysis.

Finds the optimal K allocation for each token via exhaustive search.
This provides an upper bound for learned policies and validates that:
1. Difficulty correlates with optimal K
2. Adaptive allocation can improve over uniform allocation
3. Learned policies can approach oracle performance

The oracle is computationally expensive but provides ground truth.
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

from ponderttt.data.wikitext import get_wikitext2_dataloaders
from ponderttt.models import (
    IterativeTransformerConfig,
    IterativeTransformerTTT,
)


class OracleAnalyzer:
    """
    Finds optimal K allocation via exhaustive search.
    """

    def __init__(
        self,
        model: nn.Module,
        step_options: List[int] = [1, 2, 4, 8],
        device: str = 'cuda',
    ):
        self.model = model
        self.step_options = step_options
        self.device = device

    def find_oracle_k_for_token(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        token_position: int,
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal K for a specific token position via exhaustive search.

        IMPORTANT: Due to sequential fast-weight carry-over, changing K at position t
        affects all future positions t+1, t+2, ..., T. Therefore, we measure the
        per-token loss ONLY at position t to isolate the effect of K_t.

        This provides a per-token oracle K that would be optimal if we could
        independently control each position's iteration count (which we can't in
        sequential mode, but this still provides valuable insight).

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len)
            token_position: Which token to optimize

        Returns:
            best_k: Optimal K value for this token
            losses: Dictionary mapping K → per-token loss at position t
        """
        self.model.eval()

        losses = {}

        # Try each K value
        # Note: We allow gradient computation for fast-weight updates during forward,
        # but we don't call backward()
        for k in self.step_options:
            # Create num_steps tensor with K only at target position
            # NOTE: This affects all future positions due to sequential dependency,
            # but we only measure loss at position t to get per-token oracle K
            num_steps = torch.ones(input_ids.shape[1], dtype=torch.long, device=self.device)
            num_steps[token_position] = k

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                num_steps=num_steps,
                return_stats=False,
            )

            # Get per-token loss at the target position
            # shift_logits: (batch, seq_len-1, vocab_size)
            # shift_labels: (batch, seq_len-1)
            shift_logits = outputs['logits'][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute per-token loss (no reduction)
            per_token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(shift_labels.shape)  # (batch, seq_len-1)

            # Extract loss at target position
            # Note: token_position in input corresponds to token_position-1 in shifted labels
            if token_position > 0:
                target_loss = per_token_loss[:, token_position - 1].mean().item()
            else:
                # First token has no prediction (shifted), use overall loss as fallback
                target_loss = outputs['loss'].item()

            losses[k] = target_loss

        # Find best K (lowest per-token loss)
        best_k = min(losses, key=losses.get)

        return best_k, losses

    def compute_difficulty_metrics(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        token_position: int,
    ) -> Dict[str, float]:
        """
        Compute various difficulty metrics for a token.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
            token_position: Which token

        Returns:
            metrics: Dictionary of difficulty metrics
        """
        # Entropy
        probs = torch.softmax(logits[:, token_position, :], dim=-1)
        log_probs = torch.log_softmax(logits[:, token_position, :], dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        # Per-token loss
        true_label = labels[:, token_position]
        true_log_prob = torch.gather(
            log_probs,
            dim=-1,
            index=true_label.unsqueeze(-1)
        ).squeeze(-1)
        token_loss = -true_log_prob.mean().item()

        # Gradient norm (approximate)
        grad_norm = torch.norm(hidden_states[:, token_position, :], p=2, dim=-1).mean().item()

        # Perplexity
        perplexity = np.exp(entropy)

        return {
            'entropy': entropy,
            'loss': token_loss,
            'gradient_norm': grad_norm,
            'perplexity': perplexity,
        }

    def analyze_batch(
        self,
        batch: Dict,
        sample_positions: int = 32,
    ) -> List[Dict]:
        """
        Analyze a batch and find oracle K for sampled positions.

        Args:
            batch: Batch dictionary
            sample_positions: Number of positions to sample

        Returns:
            results: List of analysis results
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        batch_size, seq_len = input_ids.shape

        # Get model outputs for difficulty metrics
        # Note: Need gradients for fast-weight updates, but don't call backward()
        outputs = self.model(
            input_ids=input_ids,
            labels=None,
            return_stats=False,
        )
        logits = outputs['logits']

        # Get hidden states (approximate with embeddings)
        with torch.no_grad():
            hidden_states = self.model.token_embedding(input_ids)

        # Sample positions to analyze
        positions = np.random.choice(
            seq_len,
            size=min(sample_positions, seq_len),
            replace=False
        )

        results = []

        for pos in positions:
            # Find oracle K
            best_k, k_losses = self.find_oracle_k_for_token(
                input_ids, labels, pos
            )

            # Compute difficulty metrics
            difficulty = self.compute_difficulty_metrics(
                hidden_states, logits, labels, pos
            )

            # Get token info
            token_id = input_ids[0, pos].item()  # First sequence
            true_label = labels[0, pos].item()

            results.append({
                'position': int(pos),
                'token_id': int(token_id),
                'true_label': int(true_label),
                'oracle_k': int(best_k),
                'k_losses': {int(k): float(loss) for k, loss in k_losses.items()},
                'difficulty': difficulty,
            })

        return results

    def analyze_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 10,
        sample_positions_per_batch: int = 32,
    ) -> Dict:
        """
        Analyze dataset and collect oracle K statistics.

        Args:
            dataloader: Data loader
            max_batches: Maximum batches to process
            sample_positions_per_batch: Positions to sample per batch

        Returns:
            aggregated_results: Analysis results
        """
        all_results = []

        pbar = tqdm(dataloader, desc="Oracle analysis", total=max_batches)

        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            batch_results = self.analyze_batch(
                batch,
                sample_positions=sample_positions_per_batch,
            )

            all_results.extend(batch_results)

            # Update progress
            if batch_idx % 5 == 0:
                avg_oracle_k = np.mean([r['oracle_k'] for r in all_results])
                pbar.set_postfix({'avg_oracle_k': f'{avg_oracle_k:.2f}'})

        # Aggregate statistics
        aggregated = self._aggregate_results(all_results)

        return aggregated

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate oracle analysis results."""
        # Oracle K distribution
        oracle_k_counts = defaultdict(int)
        for r in results:
            oracle_k_counts[r['oracle_k']] += 1

        # Difficulty by oracle K
        difficulty_by_k = defaultdict(list)
        for r in results:
            k = r['oracle_k']
            difficulty_by_k[k].append(r['difficulty'])

        # Correlations
        oracle_ks = [r['oracle_k'] for r in results]
        entropies = [r['difficulty']['entropy'] for r in results]
        losses = [r['difficulty']['loss'] for r in results]
        grad_norms = [r['difficulty']['gradient_norm'] for r in results]

        # Compute correlations
        corr_entropy = np.corrcoef(oracle_ks, entropies)[0, 1]
        corr_loss = np.corrcoef(oracle_ks, losses)[0, 1]
        corr_grad = np.corrcoef(oracle_ks, grad_norms)[0, 1]

        # Average difficulty per K
        avg_difficulty_by_k = {}
        for k in self.step_options:
            if k in difficulty_by_k:
                difficulties = difficulty_by_k[k]
                avg_difficulty_by_k[k] = {
                    'entropy': np.mean([d['entropy'] for d in difficulties]),
                    'loss': np.mean([d['loss'] for d in difficulties]),
                    'gradient_norm': np.mean([d['gradient_norm'] for d in difficulties]),
                    'count': len(difficulties),
                }

        return {
            'total_tokens_analyzed': len(results),
            'oracle_k_distribution': dict(oracle_k_counts),
            'oracle_k_percentages': {
                k: count / len(results) * 100
                for k, count in oracle_k_counts.items()
            },
            'average_oracle_k': np.mean(oracle_ks),
            'correlations': {
                'oracle_k_vs_entropy': float(corr_entropy),
                'oracle_k_vs_loss': float(corr_loss),
                'oracle_k_vs_gradient_norm': float(corr_grad),
            },
            'difficulty_by_oracle_k': {
                int(k): v for k, v in avg_difficulty_by_k.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description='Oracle K allocation analysis')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_batches', type=int, default=10,
                       help='Number of batches to analyze')
    parser.add_argument('--sample_positions', type=int, default=32,
                       help='Positions to sample per batch')
    parser.add_argument('--step_options', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--output_dir', type=str, default='experiments/results/oracle')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Oracle K Allocation Analysis")
    print("="*80)
    print(f"Step options: {args.step_options}")
    print(f"Max batches: {args.max_batches}")
    print(f"Sample positions per batch: {args.sample_positions}")
    print()

    # Load data
    _, _, test_loader = get_wikitext2_dataloaders(
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

    # Create analyzer
    analyzer = OracleAnalyzer(
        model=model,
        step_options=args.step_options,
        device=args.device,
    )

    # Run analysis
    results = analyzer.analyze_dataset(
        test_loader,
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

    print("\nAverage Difficulty by Oracle K:")
    print(f"{'K':<5} {'Count':<8} {'Entropy':<12} {'Loss':<12} {'Grad Norm':<12}")
    print("-" * 60)
    for k in sorted(results['difficulty_by_oracle_k'].keys()):
        diff = results['difficulty_by_oracle_k'][k]
        print(f"{k:<5} {diff['count']:<8} "
              f"{diff['entropy']:<12.4f} {diff['loss']:<12.4f} "
              f"{diff['gradient_norm']:<12.4f}")

    # Interpretation
    print("\n" + "="*80)
    print("Interpretation")
    print("="*80)

    avg_k = results['average_oracle_k']
    print(f"\n1. Optimal average K: {avg_k:.2f}")

    if avg_k < 2.5:
        print("   → Most tokens are easy, K=1-2 sufficient")
    elif avg_k < 5.0:
        print("   → Moderate difficulty, K=2-4 beneficial")
    else:
        print("   → High difficulty, K=4-8 needed")

    corr_entropy = results['correlations']['oracle_k_vs_entropy']
    print(f"\n2. Difficulty-K correlation (entropy): {corr_entropy:.4f}")

    if corr_entropy > 0.5:
        print("   ✓ Strong positive correlation: Difficulty predicts optimal K well")
    elif corr_entropy > 0.3:
        print("   ○ Moderate correlation: Difficulty partially predicts optimal K")
    else:
        print("   ✗ Weak correlation: Difficulty is poor predictor of optimal K")

    # Learnability analysis
    k1_pct = results['oracle_k_percentages'].get(1, 0)
    k8_pct = results['oracle_k_percentages'].get(8, 0)
    print(f"\n3. Learnability:")
    print(f"   K=1 (easy):      {k1_pct:.1f}%")
    print(f"   K=8 (very hard): {k8_pct:.1f}%")

    if abs(k1_pct - k8_pct) > 20:
        print("   ✓ Significant variation → adaptive allocation beneficial")
    else:
        print("   ○ Limited variation → uniform allocation may be sufficient")

    # Save results
    results_file = output_dir / 'oracle_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
