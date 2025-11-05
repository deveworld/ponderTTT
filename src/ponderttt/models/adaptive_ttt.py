"""
Adaptive TTT: Dynamic iteration allocation for Test-Time Training.

Implements both heuristic and learned adaptive mechanisms.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.metrics import DifficultyMetrics, IterationAllocator
from .ttt_linear import TTTLinear


class HeuristicAdaptiveTTT(nn.Module):
    """
    Heuristic Adaptive TTT using difficulty metrics.

    Allocates different numbers of gradient steps per token based on
    simple heuristics like entropy or loss.

    Args:
        base_ttt: Base TTT layer (e.g., TTTLinear)
        difficulty_metric: Which metric to use ('entropy', 'loss', 'gradient', 'combined')
        buckets: Discrete iteration counts [1, 2, 4, 8]
        thresholds: Difficulty thresholds for buckets (if None, will use auto-calibration)
        auto_calibrate: If True, automatically calibrate thresholds from data
        target_distribution: Target distribution for each bucket (e.g., [0.3, 0.4, 0.3])
        logits_fn: Optional function to compute logits for entropy
    """

    base_ttt: TTTLinear
    difficulty_metric: str
    allocator: IterationAllocator
    logits_fn: Optional[nn.Module]

    def __init__(
        self,
        base_ttt: TTTLinear,
        difficulty_metric: str = "entropy",
        buckets: list = [1, 2, 4],
        thresholds: Optional[list] = None,
        auto_calibrate: bool = False,
        target_distribution: Optional[list] = None,
        logits_fn: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.base_ttt = base_ttt
        self.difficulty_metric = difficulty_metric  # type: ignore[unresolved-attribute]
        self.allocator = IterationAllocator(  # type: ignore[unresolved-attribute]
            buckets=buckets,
            thresholds=thresholds,
            auto_calibrate=auto_calibrate,
            target_distribution=target_distribution,
        )

        # Optional: logits function for entropy calculation
        # (e.g., language model head)
        self.logits_fn = logits_fn  # type: ignore[unresolved-attribute]

    def compute_difficulty(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute difficulty scores for each token.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            logits: Optional pre-computed logits

        Returns:
            difficulty: Difficulty scores (batch, seq_len)
        """
        if self.difficulty_metric == "entropy":
            # Need logits for entropy calculation
            if logits is None and self.logits_fn is not None:
                logits = self.logits_fn(x)

            if logits is None:
                raise ValueError("Entropy metric requires logits")

            difficulty = DifficultyMetrics.entropy_based(logits, normalize=True)

        elif self.difficulty_metric == "loss":
            # Compute initial TTT loss
            # Run one step to get loss
            with torch.no_grad():
                _, stats = self.base_ttt.ttt_forward(x, num_iterations=1)
                initial_loss = stats["ttt_losses"][0]
                difficulty = torch.ones_like(x[..., 0]) * initial_loss

        elif self.difficulty_metric == "gradient":
            # Compute gradient magnitude
            x_norm = self.base_ttt.ln(x)
            x_corrupted, x_target = self.base_ttt.corrupt_input(x_norm)

            hidden = self.base_ttt.W(x_corrupted)
            loss = self.base_ttt.reconstruction_loss(hidden, x_target, x_corrupted)

            # Compute gradients
            grad = torch.autograd.grad(loss, hidden, create_graph=False)[0]
            difficulty = torch.norm(grad, dim=-1)

            # Normalize
            difficulty = (difficulty - difficulty.mean()) / (difficulty.std() + 1e-8)
            difficulty = torch.sigmoid(difficulty)

        else:
            raise ValueError(f"Unknown difficulty metric: {self.difficulty_metric}")

        return difficulty

    def forward_adaptive(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        return_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with adaptive iteration allocation.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            logits: Optional pre-computed logits for entropy
            return_stats: Whether to return detailed statistics

        Returns:
            output: Output tensor
            stats: Dictionary with statistics
        """
        batch_size, seq_len, _ = x.shape

        # 1. Compute difficulty for each token
        difficulty = self.compute_difficulty(x, logits)

        # 2. Allocate iterations per token
        iterations = self.allocator.allocate(difficulty)

        # 3. Process tokens with allocated iterations
        # For efficiency, we batch by iteration count
        outputs = torch.zeros_like(x)
        all_stats: Dict[str, Any] = {
            "difficulty": difficulty,
            "iterations": iterations,
            "distribution": self.allocator.get_distribution(iterations),
            "per_token_stats": [],
        }

        # Group tokens by iteration count for efficient processing
        for bucket in self.allocator.buckets:
            mask = iterations == bucket

            if not mask.any():
                continue

            # Process tokens in this bucket
            # Note: In practice, we'd want to batch this more efficiently
            # For now, process sequence-by-sequence
            for b in range(batch_size):
                token_mask = mask[b]
                if not token_mask.any():
                    continue

                # Extract tokens for this bucket
                tokens_to_process = x[b : b + 1, token_mask, :]

                # Run TTT with this iteration count
                output, stats = self.base_ttt.ttt_forward(tokens_to_process, num_iterations=bucket)

                # Store output
                outputs[b, token_mask, :] = output[0]

                # Store stats
                if return_stats:
                    all_stats["per_token_stats"].append(
                        {
                            "batch": b,
                            "bucket": bucket,
                            "num_tokens": token_mask.sum().item(),
                            "stats": stats,
                        }
                    )

        # Compute efficiency statistics
        if return_stats:
            avg_iterations = iterations.float().mean().item()
            baseline_iterations = self.base_ttt.num_iterations

            all_stats["efficiency"] = {
                "avg_iterations": avg_iterations,
                "baseline_iterations": baseline_iterations,
                "flops_ratio": avg_iterations / baseline_iterations,
                "flops_reduction": 1.0 - (avg_iterations / baseline_iterations),
            }

        return outputs, all_stats

    def forward(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simplified forward (output only)."""
        output, _ = self.forward_adaptive(x, logits, return_stats=False)
        return output


class AdaptiveTTT(nn.Module):
    """
    Learned Adaptive TTT (Phase 2).

    Uses a neural network to predict optimal number of iterations.
    To be implemented in Phase 2.
    """

    def __init__(self, base_ttt: TTTLinear):
        super().__init__()
        self.base_ttt = base_ttt
        # TODO: Implement difficulty predictor network
        raise NotImplementedError("Learned adaptive TTT coming in Phase 2!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Coming in Phase 2!")
