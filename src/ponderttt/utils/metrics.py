"""
Difficulty metrics for adaptive iteration allocation.

Provides various ways to measure token difficulty:
- Prediction entropy
- Initial TTT loss
- Gradient magnitude
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of probability distribution.

    Args:
        logits: Logits tensor (..., vocab_size)
        dim: Dimension to compute entropy over

    Returns:
        entropy: Entropy values (...)
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


class DifficultyMetrics:
    """
    Compute various difficulty metrics for tokens.

    These metrics are used to determine how many gradient steps
    each token should receive during TTT.
    """

    @staticmethod
    def entropy_based(logits: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute difficulty based on prediction entropy.

        High entropy = uncertain prediction = difficult token

        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            normalize: Whether to normalize to [0, 1]

        Returns:
            difficulty: Difficulty scores (batch, seq_len)
        """
        entropy = compute_entropy(logits, dim=-1)

        if normalize:
            # Normalize by max possible entropy
            vocab_size = logits.size(-1)
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
            entropy = entropy / max_entropy

        return entropy

    @staticmethod
    def loss_based(loss_per_token: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute difficulty based on initial TTT loss.

        High loss = poor reconstruction = difficult token

        Args:
            loss_per_token: Loss values per token (batch, seq_len)
            normalize: Whether to normalize to [0, 1]

        Returns:
            difficulty: Difficulty scores (batch, seq_len)
        """
        difficulty = loss_per_token

        if normalize:
            # Normalize by batch statistics
            difficulty = (difficulty - difficulty.mean()) / (difficulty.std() + 1e-8)
            difficulty = torch.sigmoid(difficulty)  # Map to [0, 1]

        return difficulty

    @staticmethod
    def gradient_based(gradients: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Compute difficulty based on gradient magnitude.

        Large gradients = far from optimum = difficult token

        Args:
            gradients: Gradient tensor (batch, seq_len, ...)
            normalize: Whether to normalize to [0, 1]

        Returns:
            difficulty: Difficulty scores (batch, seq_len)
        """
        # Compute gradient norm per token
        grad_norm: torch.Tensor = torch.norm(gradients.flatten(2), dim=-1)

        if normalize:
            # Normalize by batch statistics
            grad_norm = (grad_norm - grad_norm.mean()) / (grad_norm.std() + 1e-8)
            grad_norm = torch.sigmoid(grad_norm)

        return grad_norm

    @staticmethod
    def combined(
        logits: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Combine multiple difficulty metrics.

        Args:
            logits: Model logits (for entropy)
            loss: Loss values per token
            gradients: Gradient tensor
            weights: Dict with keys 'entropy', 'loss', 'gradient'

        Returns:
            difficulty: Combined difficulty scores
        """
        if weights is None:
            weights = {"entropy": 0.4, "loss": 0.3, "gradient": 0.3}

        difficulties = []
        weight_sum = 0.0

        if logits is not None and "entropy" in weights:
            diff = DifficultyMetrics.entropy_based(logits, normalize=True)
            difficulties.append(weights["entropy"] * diff)
            weight_sum += weights["entropy"]

        if loss is not None and "loss" in weights:
            diff = DifficultyMetrics.loss_based(loss, normalize=True)
            difficulties.append(weights["loss"] * diff)
            weight_sum += weights["loss"]

        if gradients is not None and "gradient" in weights:
            diff = DifficultyMetrics.gradient_based(gradients, normalize=True)
            difficulties.append(weights["gradient"] * diff)
            weight_sum += weights["gradient"]

        if not difficulties:
            raise ValueError("At least one difficulty metric must be provided")

        # Weighted average
        combined = torch.stack(difficulties).sum(dim=0) / weight_sum

        return combined


class IterationAllocator:
    """
    Allocate number of iterations based on difficulty scores.

    Uses discrete buckets for hardware efficiency.
    """

    buckets: list
    num_buckets: int
    auto_calibrate: bool
    target_distribution: Optional[list]
    thresholds: Optional[list]
    calibrated: bool

    def __init__(
        self,
        buckets: list = [1, 2, 4],
        thresholds: Optional[list] = None,
        auto_calibrate: bool = False,
        target_distribution: Optional[list] = None,
    ):
        """
        Args:
            buckets: Discrete iteration counts (e.g., [1, 2, 4])
            thresholds: Difficulty thresholds for each bucket
                       If None, will use equal spacing
            auto_calibrate: If True, thresholds will be calibrated from data
            target_distribution: Target fraction for each bucket (e.g., [0.3, 0.4, 0.3])
                                Only used if auto_calibrate is True
        """
        self.buckets = sorted(buckets)
        self.num_buckets = len(buckets)
        self.auto_calibrate = auto_calibrate

        if target_distribution is None:
            # Default: roughly equal distribution with slight bias toward middle
            target_distribution = [0.3, 0.4, 0.3] if self.num_buckets == 3 else None
            if self.num_buckets != 3 and target_distribution is None:
                target_distribution = [1.0 / self.num_buckets] * self.num_buckets

        self.target_distribution = target_distribution

        if thresholds is None and not auto_calibrate:
            # Equal spacing between 0 and 1
            thresholds = [i / self.num_buckets for i in range(1, self.num_buckets)]

        self.thresholds = thresholds if thresholds is not None else None
        self.calibrated = not auto_calibrate

    def calibrate(self, difficulty_samples: torch.Tensor) -> None:
        """
        Calibrate thresholds based on difficulty samples using percentile method.

        This ensures the actual distribution matches the target distribution.

        Args:
            difficulty_samples: Sample difficulty scores to calibrate on
                               Shape: (num_samples,) or (batch, seq_len)

        Example:
            For target_distribution = [0.3, 0.4, 0.3] (30% easy, 40% med, 30% hard):
            - Threshold 1 at 30th percentile (separates easy from medium)
            - Threshold 2 at 70th percentile (separates medium from hard)
        """
        # Flatten samples
        samples = difficulty_samples.flatten()

        # Compute cumulative percentiles for thresholds
        cumulative_dist = torch.tensor(self.target_distribution).cumsum(0)[:-1]
        percentiles = (cumulative_dist * 100).tolist()

        # Compute thresholds using percentiles
        self.thresholds = []
        for percentile in percentiles:
            threshold = torch.quantile(samples, percentile / 100.0).item()
            self.thresholds.append(threshold)

        self.calibrated = True

    def calibrate_multi_batch(self, difficulty_batches: list) -> None:
        """
        Calibrate thresholds using multiple batches for improved stability.

        Args:
            difficulty_batches: List of difficulty tensors from multiple batches
        """
        all_samples = torch.cat([d.flatten() for d in difficulty_batches], dim=0)
        self.calibrate(all_samples)

    def allocate(
        self, difficulty: torch.Tensor, auto_calibrate: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Allocate iterations based on difficulty.

        Args:
            difficulty: Difficulty scores (batch, seq_len)
            auto_calibrate: If True, calibrate thresholds on-the-fly from this batch
                           If None, use instance setting

        Returns:
            iterations: Number of iterations per token (batch, seq_len)
        """
        if auto_calibrate is None:
            auto_calibrate = self.auto_calibrate

        if auto_calibrate and not self.calibrated and self.thresholds is None:
            self.calibrate(difficulty)
            self.calibrated = True

        if self.thresholds is None:
            raise ValueError(
                "Thresholds not set. Call calibrate() first or set auto_calibrate=True"
            )

        iterations: torch.Tensor = torch.ones_like(difficulty, dtype=torch.long) * self.buckets[0]

        for i, threshold in enumerate(self.thresholds):
            mask = difficulty > threshold
            iterations[mask] = self.buckets[i + 1]

        return iterations

    def freeze_calibration(self) -> None:
        """
        Freeze calibration to prevent further updates.

        Call this after training is complete, before evaluation on test set,
        to prevent test data leakage.
        """
        if not self.calibrated:
            raise ValueError(
                "Cannot freeze calibration before it has been performed. "
                "Call calibrate() with validation data first."
            )
        self.auto_calibrate = False

    def get_calibration_status(self) -> dict:
        """
        Get current calibration status.

        Returns:
            Dictionary with calibration information
        """
        return {
            "calibrated": self.calibrated,
            "auto_calibrate": self.auto_calibrate,
            "thresholds": self.thresholds,
            "buckets": self.buckets,
            "target_distribution": self.target_distribution,
        }

    def get_distribution(self, iterations: torch.Tensor) -> Dict[int, float]:
        """
        Get distribution of allocated iterations.

        Args:
            iterations: Iteration allocations (batch, seq_len)

        Returns:
            distribution: Dict mapping bucket -> fraction
        """
        total = iterations.numel()
        distribution = {}

        for bucket in self.buckets:
            count = (iterations == bucket).sum().item()
            distribution[bucket] = count / total

        return distribution


def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.

    Args:
        loss: Cross-entropy loss (scalar)

    Returns:
        perplexity: exp(loss)
    """
    return math.exp(min(loss, 100))  # Clip for numerical stability
