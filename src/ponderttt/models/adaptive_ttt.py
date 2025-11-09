"""
Adaptive TTT: Per-token learning rate modulation for Test-Time Training.

Implements heuristic-based adaptive learning rate scaling using difficulty metrics.
"""

from typing import Any, Dict, Optional, Tuple, Union
import time

import torch
import torch.nn as nn

from ..utils.metrics import DifficultyMetrics, IterationAllocator
from .ttt_linear import TTTLinear, TTTLinearSequential, ln_fwd


class HeuristicAdaptiveTTT(nn.Module):
    """
    Heuristic Adaptive TTT using difficulty metrics.

    Scales per-token learning rates based on difficulty metrics
    such as entropy, loss, or gradient magnitude.

    Args:
        base_ttt: Base TTT layer (e.g., TTTLinear)
        difficulty_metric: Which metric to use ('entropy', 'loss', 'gradient', 'combined')
        buckets: Discrete iteration counts [1, 2, 4, 8]
        thresholds: Difficulty thresholds for buckets (if None, will use auto-calibration)
        auto_calibrate: If True, automatically calibrate thresholds from data
        target_distribution: Target distribution for each bucket (e.g., [0.3, 0.4, 0.3])
        logits_fn: Optional function to compute logits for entropy
    """

    def __init__(
        self,
        base_ttt: Union[TTTLinear, TTTLinearSequential],
        difficulty_metric: str = "entropy",
        buckets: list = [1, 2, 4],
        thresholds: Optional[list] = None,
        auto_calibrate: bool = False,
        target_distribution: Optional[list] = None,
        logits_fn: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.base_ttt: Union[TTTLinear, TTTLinearSequential] = base_ttt
        self.difficulty_metric: str = difficulty_metric
        self.allocator: IterationAllocator = IterationAllocator(
            buckets=buckets,
            thresholds=thresholds,
            auto_calibrate=auto_calibrate,
            target_distribution=target_distribution,
        )

        # Optional: logits function for entropy calculation
        # (e.g., language model head)
        self.logits_fn: Optional[nn.Module] = logits_fn

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, L, hidden) tensor into (B, num_heads, L, head_dim)."""
        batch, seq_len, _ = tensor.shape
        num_heads = self.base_ttt.num_heads
        head_dim = self.base_ttt.head_dim
        return tensor.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states through the analytic TTT projections."""
        XQ, XK, XV = self.base_ttt.get_qkv(x)
        return self._reshape_heads(XQ), self._reshape_heads(XK), self._reshape_heads(XV)

    def _estimate_loss_and_gradients(
        self,
        x: torch.Tensor,
        cached_qkv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate per-token reconstruction loss and residual gradients without autograd.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            cached_qkv: Optional pre-computed (XQ_heads, XK_heads, XV_heads) to avoid recomputation

        Returns:
            loss_per_token: Per-token reconstruction loss
            gradients: Per-token gradient magnitudes
        """
        with torch.no_grad():
            if cached_qkv is not None:
                _, XK_heads, XV_heads = cached_qkv
            else:
                _, XK_heads, XV_heads = self._project_qkv(x)

            Z = torch.einsum("bnlf,nfd->bnld", XK_heads, self.base_ttt.W1)
            Z = Z + self.base_ttt.b1.unsqueeze(0)

            gamma = self.base_ttt.ttt_norm_weight.reshape(self.base_ttt.num_heads, 1, self.base_ttt.head_dim)
            beta = self.base_ttt.ttt_norm_bias.reshape(self.base_ttt.num_heads, 1, self.base_ttt.head_dim)
            Z_norm = ln_fwd(Z, gamma, beta)

            reconstruction_target = XV_heads - XK_heads
            residual = Z_norm - reconstruction_target

            loss_per_token = residual.pow(2).mean(dim=-1).mean(dim=1)

            batch, seq_len = x.shape[:2]
            gradients = residual.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        return loss_per_token, gradients

    def compute_difficulty(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        cached_qkv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute difficulty scores for each token.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            logits: Optional pre-computed logits
            cached_qkv: Optional pre-computed (XQ_heads, XK_heads, XV_heads) to avoid recomputation

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
            loss_per_token, _ = self._estimate_loss_and_gradients(x, cached_qkv=cached_qkv)
            difficulty = DifficultyMetrics.loss_based(loss_per_token, normalize=True)

        elif self.difficulty_metric == "gradient":
            _, gradients = self._estimate_loss_and_gradients(x, cached_qkv=cached_qkv)
            difficulty = DifficultyMetrics.gradient_based(gradients, normalize=True)

        elif self.difficulty_metric == "combined":
            loss_per_token, gradients = self._estimate_loss_and_gradients(x, cached_qkv=cached_qkv)
            difficulty = DifficultyMetrics.combined(
                logits=logits,
                loss=loss_per_token,
                gradients=gradients,
            )

        else:
            raise ValueError(f"Unknown difficulty metric: {self.difficulty_metric}")

        return difficulty.detach()

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
        start_time = time.time()
        batch_size, seq_len, _ = x.shape

        cached_qkv = None
        if self.difficulty_metric in {"loss", "gradient", "combined"}:
            with torch.no_grad():
                cached_qkv = self._project_qkv(x)

        difficulty = self.compute_difficulty(x, logits, cached_qkv=cached_qkv)
        iterations = self.allocator.allocate(difficulty)

        max_iter = max(self.allocator.buckets)
        scaling = iterations / max_iter

        outputs, base_stats = self.base_ttt.ttt_forward(x, token_scaling=scaling)

        all_stats: Dict[str, Any] = {
            "difficulty": difficulty.detach(),
            "iterations": iterations.detach(),
            "distribution": self.allocator.get_distribution(iterations),
        }
        if return_stats and base_stats:
            all_stats.update(base_stats)

        if return_stats:
            all_stats["avg_scaling"] = float(scaling.mean().item())

        all_stats["wall_clock_time"] = time.time() - start_time
        return outputs, all_stats

    def forward(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simplified forward (output only)."""
        output, _ = self.forward_adaptive(x, logits, return_stats=False)
        return output

    def calibrate_on_dataloader(
        self,
        dataloader,
        num_batches: int = 10,
        device: str = "cpu",
        embedding_fn: Optional[nn.Module] = None,
        logits_fn: Optional[nn.Module] = None,
    ) -> None:
        """
        Calibrate thresholds using multiple batches from dataloader for stability.

        Args:
            dataloader: DataLoader or iterable producing batches
            num_batches: Number of batches to use for calibration
            device: Device to use for computation
            embedding_fn: Optional function to convert input_ids to embeddings
                         (Required if dataloader produces input_ids instead of hidden_states)
            logits_fn: Optional function to compute logits for entropy metric
                      (Required if difficulty_metric is 'entropy' or 'combined')

        Example:
            # For a full model:
            model.adaptive_ttt.calibrate_on_dataloader(
                dataloader,
                num_batches=10,
                device='cuda',
                embedding_fn=lambda ids: model.token_embedding(ids) + model.position_embedding(...),
                logits_fn=lambda h: model.lm_head(model.ln_f(h))
            )
        """
        self.eval()
        difficulty_batches = []

        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            # Extract tensors from batch
            if isinstance(batch, dict):
                if "hidden_states" in batch:
                    # Batch already contains hidden states
                    hidden_states = batch["hidden_states"].to(device)
                elif "input_ids" in batch:
                    # Need to embed input_ids
                    if embedding_fn is None:
                        raise ValueError(
                            "Calibration batch contains 'input_ids' but no embedding_fn provided. "
                            "Please provide embedding_fn parameter to convert input_ids to hidden states."
                        )
                    input_ids = batch["input_ids"].to(device)
                    hidden_states = embedding_fn(input_ids)
                else:
                    raise ValueError("Calibration batch must contain 'hidden_states' or 'input_ids'")
            else:
                # Assume batch is already hidden states tensor
                hidden_states = batch.to(device)

            # Compute logits if needed for entropy metric
            logits = None
            if self.difficulty_metric in {"entropy", "combined"}:
                if logits_fn is not None:
                    logits = logits_fn(hidden_states)
                elif self.logits_fn is not None:
                    logits = self.logits_fn(hidden_states)
                else:
                    raise ValueError(
                        f"Difficulty metric '{self.difficulty_metric}' requires logits, "
                        "but no logits_fn provided"
                    )

            difficulty = self.compute_difficulty(hidden_states, logits=logits).detach()
            difficulty_batches.append(difficulty.cpu())
        torch.set_grad_enabled(prev_grad_state)

        self.allocator.calibrate_multi_batch(difficulty_batches)
        print(f"Calibrated on {len(difficulty_batches)} batches")

    def freeze_calibration(self) -> None:
        """
        Freeze calibration after training to prevent test data leakage.

        Should be called after training is complete, before test evaluation.
        """
        self.allocator.freeze_calibration()

    def get_calibration_status(self) -> dict:
        """Get current calibration status."""
        return self.allocator.get_calibration_status()


class AdaptiveTTT(nn.Module):
    """
    Learned Adaptive TTT (Phase 2).

    Uses a neural network to predict optimal number of iterations.
    To be implemented in Phase 2.
    """

    def __init__(self, base_ttt: Union[TTTLinear, TTTLinearSequential]):
        super().__init__()
        self.base_ttt = base_ttt
        # TODO: Implement difficulty predictor network
        raise NotImplementedError("Learned adaptive TTT coming in Phase 2!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Coming in Phase 2!")
