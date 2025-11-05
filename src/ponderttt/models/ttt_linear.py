"""
TTT-Linear: Test-Time Training with Linear Hidden State

Core implementation of TTT layer with linear model as hidden state.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTLinear(nn.Module):
    """
    TTT-Linear layer with fixed number of gradient descent steps.

    The hidden state is a linear model W that is updated via gradient descent
    on a self-supervised reconstruction task at test time.

    Args:
        hidden_dim: Dimension of input/output tokens
        ttt_dim: Dimension of TTT hidden state (inner model)
        num_iterations: Fixed number of gradient steps (default: 2)
        learning_rate: Learning rate for inner loop optimization
        reconstruction_type: Type of reconstruction task ('masked', 'autoencoder')
        mask_ratio: Ratio of features to mask for reconstruction
    """

    hidden_dim: int
    ttt_dim: int
    num_iterations: int
    learning_rate: float
    reconstruction_type: str
    mask_ratio: float

    def __init__(
        self,
        hidden_dim: int,
        ttt_dim: int = 256,
        num_iterations: int = 2,
        learning_rate: float = 0.01,
        reconstruction_type: str = "masked",
        mask_ratio: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim  # type: ignore[unresolved-attribute]
        self.ttt_dim = ttt_dim  # type: ignore[unresolved-attribute]
        self.num_iterations = num_iterations  # type: ignore[unresolved-attribute]
        self.learning_rate = learning_rate  # type: ignore[unresolved-attribute]
        self.reconstruction_type = reconstruction_type  # type: ignore[unresolved-attribute]
        self.mask_ratio = mask_ratio  # type: ignore[unresolved-attribute]

        # Inner model: Linear layer (the "hidden state" in TTT)
        # This gets updated via gradient descent at test time
        self.W = nn.Linear(hidden_dim, ttt_dim, bias=False)

        # Reconstruction head
        self.reconstruction_head = nn.Linear(ttt_dim, hidden_dim, bias=False)

        # Layer norm (applied before TTT)
        self.ln = nn.LayerNorm(hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(ttt_dim, hidden_dim)

    def corrupt_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupt input for self-supervised reconstruction.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)

        Returns:
            corrupted: Corrupted input
            target: Original input (target for reconstruction)
        """
        if self.reconstruction_type == "masked":
            # Random masking
            batch_size, seq_len, dim = x.shape
            mask = torch.rand(batch_size, seq_len, dim, device=x.device) > self.mask_ratio
            corrupted = x * mask.float()
            return corrupted, x

        elif self.reconstruction_type == "autoencoder":
            # Simple autoencoder (no corruption)
            return x, x

        else:
            raise ValueError(f"Unknown reconstruction type: {self.reconstruction_type}")

    def reconstruction_loss(
        self, hidden: torch.Tensor, target: torch.Tensor, corrupted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            hidden: Hidden representation from W(corrupted)
            target: Original input (ground truth)
            corrupted: Corrupted input

        Returns:
            loss: Reconstruction loss
        """
        # Reconstruct from hidden
        reconstructed = self.reconstruction_head(hidden)

        # MSE loss
        loss = F.mse_loss(reconstructed, target)

        return loss

    def ttt_forward(
        self,
        x: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with Test-Time Training.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            num_iterations: Number of gradient steps (overrides default if provided)

        Returns:
            output: Output tensor
            stats: Dictionary with statistics (losses, etc.)
        """
        num_iters = num_iterations if num_iterations is not None else self.num_iterations

        # Normalize input
        x_norm = self.ln(x)

        # Corrupt input for reconstruction
        x_corrupted, x_target = self.corrupt_input(x_norm)

        # Store initial weights
        W_init = self.W.weight.data.clone()

        # Track losses
        losses = []

        # Test-Time Training: Gradient descent on reconstruction loss
        # We need gradients enabled for TTT even in eval mode
        with torch.enable_grad():
            for step in range(num_iters):
                # Ensure W requires grad
                self.W.weight.requires_grad_(True)

                # Forward through inner model
                hidden = self.W(x_corrupted)

                # Compute reconstruction loss
                loss = self.reconstruction_loss(hidden, x_target, x_corrupted)
                losses.append(loss.item())

                # Gradient descent step (manual update)
                if step < num_iters - 1:  # Don't update after last iteration
                    # Compute gradients
                    grad_W = torch.autograd.grad(
                        loss, self.W.weight, create_graph=False, retain_graph=False
                    )[0]

                    # Update weights (SGD step)
                    with torch.no_grad():
                        self.W.weight.data -= self.learning_rate * grad_W

        # Final forward pass with updated weights
        with torch.no_grad():
            hidden_final = self.W(x_norm)
            output = self.out_proj(hidden_final)

        # Reset weights to initial state (important for next token!)
        with torch.no_grad():
            self.W.weight.data.copy_(W_init)

        # Statistics
        stats = {
            "ttt_losses": losses,
            "num_iterations": num_iters,
            "final_loss": losses[-1] if losses else 0.0,
        }

        # Add residual connection
        output = output + x

        return output, stats

    def forward(
        self,
        x: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass (simplified interface).

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            num_iterations: Number of gradient steps

        Returns:
            output: Output tensor
        """
        output, _ = self.ttt_forward(x, num_iterations)
        return output


class TTTLinearWithStats(TTTLinear):
    """
    TTT-Linear that returns both output and statistics.
    Useful for debugging and analysis.
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        num_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass returning output and stats."""
        return self.ttt_forward(x, num_iterations)
