"""
Official-style TTT Layer with Analytic Update.

Simplified PyTorch implementation of the official TTT layer from:
https://github.com/test-time-training/ttt-lm-jax

Key differences from iterative TTT:
- Analytic closed-form update (not iterative gradient descent)
- Mini-batch processing (16 tokens per mini-batch)
- Linear fast weights (W @ K + b, not MLP)
- Position-dependent learning rate modulation
- Single update per mini-batch (not K-step loops)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTNorm(nn.Module):
    """
    TTT normalization layer.

    Simple LayerNorm wrapper for compatibility with official implementation.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class LearnableLRNetwork(nn.Module):
    """
    Learnable learning rate modulation network.

    Predicts per-token LR scaling factors from hidden states.
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class OfficialTTTLayer(nn.Module):
    """
    Official-style TTT layer with analytic update.

    Implements the closed-form solution from the official JAX implementation:
    - W1_bar = W1_init - (eta * X1)^T @ grad_l_wrt_Z1
    - b1_bar = b1_init - sum(eta * grad_l_wrt_Z1)

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        mini_batch_size: Size of each mini-batch (typically 16)
        base_lr: Base learning rate for TTT updates
        use_learnable_lr: Whether to use learnable LR modulation
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        mini_batch_size: int = 16,
        base_lr: float = 0.1,
        use_learnable_lr: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        assert hidden_dim == num_heads * self.head_dim

        self.mini_batch_size = mini_batch_size
        self.base_lr = base_lr
        self.use_learnable_lr = use_learnable_lr

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fast-weight parameters (linear: W @ K + b)
        # Per-head parameters
        self.W1_init = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.b1_init = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))

        # TTT normalization
        self.ttt_norm = TTTNorm(self.head_dim)

        # Learnable LR modulation (optional)
        if use_learnable_lr:
            self.lr_network = LearnableLRNetwork(hidden_dim, mini_batch_size)
        else:
            self.lr_network = None

        # Position-dependent token index
        self.register_buffer(
            'token_idx',
            1.0 / torch.arange(1, mini_batch_size + 1, dtype=torch.float32)
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)

        # Optional gating
        self.use_gate = True
        if self.use_gate:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.use_gate:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

    def _get_eta(
        self,
        x: torch.Tensor,
        mini_batch_size: int,
    ) -> torch.Tensor:
        """
        Compute position-dependent learning rates.

        Args:
            x: Input (batch, seq_len, hidden_dim)
            mini_batch_size: Effective mini-batch size

        Returns:
            eta: Learning rates (batch, num_heads, mini_batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Token position indices (1/16, 2/16, ..., 16/16)
        token_idx = self.token_idx[:mini_batch_size].view(1, 1, mini_batch_size, 1)

        # Base LR scaled by position
        eta = self.base_lr * token_idx / self.head_dim

        # Optional learnable modulation
        if self.lr_network is not None:
            # Compute modulation from input
            lr_mod = self.lr_network(x[:, :mini_batch_size, :])  # (batch, mini_batch_size, 1)
            lr_mod = lr_mod.view(batch_size, 1, mini_batch_size, 1)
            eta = eta * lr_mod

        # Broadcast to (batch, num_heads, mini_batch_size, 1)
        eta = eta.expand(batch_size, self.num_heads, mini_batch_size, 1)

        return eta

    def _process_mini_batch(
        self,
        q_mb: torch.Tensor,
        k_mb: torch.Tensor,
        v_mb: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        eta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single mini-batch with analytic update and triangular attention.

        Args:
            q_mb: Query mini-batch (batch, num_heads, mb_size, head_dim)
            k_mb: Key mini-batch (batch, num_heads, mb_size, head_dim)
            v_mb: Value mini-batch (batch, num_heads, mb_size, head_dim)
            W1: Current fast-weight matrix (batch, num_heads, head_dim, head_dim)
            b1: Current fast-weight bias (batch, num_heads, 1, head_dim)
            eta: Learning rates (batch, num_heads, mb_size, 1)

        Returns:
            output_mb: Output for this mini-batch
            W1_new: Updated fast-weight matrix
            b1_new: Updated fast-weight bias
        """
        batch_size, num_heads, mb_size, head_dim = k_mb.shape

        # Forward pass through current fast-weights
        # Z1 = K @ W1 + b1
        Z1 = torch.matmul(k_mb, W1) + b1  # (batch, num_heads, mb_size, head_dim)

        # Apply normalization
        Z1_norm = self.ttt_norm(Z1)

        # SSL target: V - K
        ssl_target = v_mb - k_mb

        # Compute gradient of loss w.r.t. Z1
        grad_loss_wrt_norm = Z1_norm - ssl_target

        # Backprop through norm (simplified: treat as identity for now)
        grad_l_wrt_Z1 = grad_loss_wrt_norm

        # Analytic update (closed-form solution)
        # W1_bar = W1 - eta * X1^T @ grad_l_wrt_Z1
        # Use last eta for the update
        last_eta = eta[:, :, -1:, :]  # (batch, num_heads, 1, 1)

        # W1_new = W1 - (last_eta * K)^T @ grad
        grad_W = torch.matmul(
            k_mb.transpose(-2, -1),  # (batch, num_heads, head_dim, mb_size)
            grad_l_wrt_Z1  # (batch, num_heads, mb_size, head_dim)
        )  # (batch, num_heads, head_dim, head_dim)
        W1_new = W1 - last_eta.squeeze(-1).unsqueeze(-1) * grad_W

        # b1_new = b1 - sum(last_eta * grad)
        grad_b = (last_eta * grad_l_wrt_Z1).sum(dim=2, keepdim=True)
        b1_new = b1 - grad_b

        # Compute output using triangular attention (causal)
        # X1_bar = Q, X1 = K
        # Attn1 = tril(Q @ K^T)
        X1_bar = q_mb  # (batch, num_heads, mb_size, head_dim)
        X1 = k_mb      # (batch, num_heads, mb_size, head_dim)

        # Compute attention scores
        # (batch, num_heads, mb_size, head_dim) @ (batch, num_heads, head_dim, mb_size)
        attn_scores = torch.matmul(X1_bar, X1.transpose(-2, -1))  # (batch, num_heads, mb_size, mb_size)

        # Apply lower triangular mask (causal)
        Attn1 = torch.tril(attn_scores)  # (batch, num_heads, mb_size, mb_size)

        # Square eta for the update
        square_eta_mini_batch = eta * eta  # (batch, num_heads, mb_size, 1)

        # Expand to (batch, num_heads, mb_size, mb_size) for matrix operations
        square_eta_expanded = square_eta_mini_batch.squeeze(-1).unsqueeze(-1)  # (batch, num_heads, mb_size, 1)

        # Create triangular mask for bias update
        tril_mask = torch.tril(torch.ones(mb_size, mb_size, device=q_mb.device, dtype=q_mb.dtype))
        tril_mask = tril_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, mb_size, mb_size)

        # b1_bar = b1_init - (square_eta * tril(ones)) @ grad
        # Shape: (batch, num_heads, mb_size, 1) * (1, 1, mb_size, mb_size) @ (batch, num_heads, mb_size, head_dim)
        b1_bar = b1 - torch.matmul(square_eta_expanded * tril_mask, grad_l_wrt_Z1)

        # Z1_bar = Q @ W1_init - (square_eta * Attn1) @ grad + b1_bar
        # (batch, num_heads, mb_size, head_dim) @ (batch, num_heads, head_dim, head_dim)
        Z1_bar = torch.matmul(X1_bar, W1)
        # - (batch, num_heads, mb_size, mb_size) * (batch, num_heads, mb_size, 1) @ (batch, num_heads, mb_size, head_dim)
        Z1_bar = Z1_bar - torch.matmul(square_eta_expanded * Attn1, grad_l_wrt_Z1)
        Z1_bar = Z1_bar + b1_bar

        # Apply normalization to output
        ttt_norm_out_bar = self.ttt_norm(Z1_bar)

        # Final output
        output_mb = X1_bar + ttt_norm_out_bar

        return output_mb, W1_new, b1_new

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, hidden_dim)
            return_stats: Whether to return statistics

        Returns:
            output: Output (batch, seq_len, hidden_dim)
            stats: Optional statistics
        """
        batch_size, seq_len, _ = x.shape

        # Pad sequence to multiple of mini_batch_size
        n_mini_batches = (seq_len + self.mini_batch_size - 1) // self.mini_batch_size
        padded_len = n_mini_batches * self.mini_batch_size

        if padded_len > seq_len:
            padding = torch.zeros(
                batch_size, padded_len - seq_len, self.hidden_dim,
                device=x.device, dtype=x.dtype
            )
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        # Project to Q, K, V
        q = self.q_proj(x_padded)
        k = self.k_proj(x_padded)
        v = self.v_proj(x_padded)

        # Reshape to multi-head
        q = q.view(batch_size, padded_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, padded_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, padded_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Initialize fast-weights (broadcast to batch dimension)
        W1 = self.W1_init.unsqueeze(0).expand(batch_size, -1, -1, -1)
        b1 = self.b1_init.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Process each mini-batch sequentially
        outputs = []

        for i in range(n_mini_batches):
            start_idx = i * self.mini_batch_size
            end_idx = start_idx + self.mini_batch_size

            q_mb = q[:, :, start_idx:end_idx, :]
            k_mb = k[:, :, start_idx:end_idx, :]
            v_mb = v[:, :, start_idx:end_idx, :]

            # Compute eta for this mini-batch
            eta = self._get_eta(
                x_padded[:, start_idx:end_idx, :],
                mini_batch_size=q_mb.shape[2]
            )

            # Process mini-batch
            output_mb, W1, b1 = self._process_mini_batch(
                q_mb, k_mb, v_mb, W1, b1, eta
            )

            outputs.append(output_mb)

        # Concatenate outputs
        output = torch.cat(outputs, dim=2)  # (batch, num_heads, padded_len, head_dim)

        # Remove padding
        output = output[:, :, :seq_len, :]

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Layer normalization
        output = self.ln(output)

        # Optional gating
        if self.use_gate:
            gate = self.gate_proj(x)
            gate = F.gelu(gate, approximate="tanh")
            output = gate * output

        # Output projection
        output = self.out_proj(output)

        stats = None
        if return_stats:
            stats = {
                'method': 'official_analytic',
                'n_mini_batches': n_mini_batches,
                'mini_batch_size': self.mini_batch_size,
            }

        return output, stats
