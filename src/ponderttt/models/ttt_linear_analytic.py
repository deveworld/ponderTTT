"""
Official TTT-Linear implementation with analytic solution.

Based on: https://github.com/test-time-training/ttt-lm-jax
Paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"

Key differences from iterative variant:
1. Analytic closed-form solution (not iterative GD)
2. Mini-batch processing (16 tokens at a time)
3. Triangular attention within mini-batch
4. Per-head learnable learning rate network
5. Two-stage LayerNorm (before and after residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def layernorm_vjp(x: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute vector-Jacobian product for LayerNorm.

    This implements the backward pass of LayerNorm to properly backpropagate
    gradients through the normalization, as done in the official JAX implementation.

    Args:
        x: Input to LayerNorm (*, hidden_dim)
        grad_output: Gradient w.r.t. LayerNorm output (*, hidden_dim)
        weight: LayerNorm weight (hidden_dim,)
        bias: LayerNorm bias (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        grad_input: Gradient w.r.t. LayerNorm input (*, hidden_dim)
    """
    # Compute normalization statistics
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)

    # Normalized input
    x_norm = (x - mean) / std

    # Gradient computation (standard LayerNorm backward)
    # grad_input = weight * (grad_output - grad_output.mean() - x_norm * (grad_output * x_norm).mean()) / std

    N = x.shape[-1]
    grad_x_norm = grad_output * weight

    # Backward through normalization
    grad_var = (grad_x_norm * (x - mean)).sum(dim=-1, keepdim=True) * (-0.5) * (var + eps) ** (-1.5)
    grad_mean = (grad_x_norm * (-1.0 / std)).sum(dim=-1, keepdim=True) + grad_var * (-2.0 * (x - mean)).sum(dim=-1, keepdim=True) / N

    grad_input = grad_x_norm / std + grad_var * 2.0 * (x - mean) / N + grad_mean / N

    return grad_input


class TTTLinearAnalytic(nn.Module):
    """
    Official TTT-Linear layer with analytic solution.

    Implements the exact algorithm from the paper, including:
    - Mini-batch processing (configurable size, default 16)
    - Triangular attention for causal dependencies
    - Analytic gradient descent (closed-form update)
    - Learnable per-head learning rates
    - Proper two-stage normalization

    Args:
        hidden_dim: Total hidden dimension
        num_heads: Number of attention heads
        mini_batch_size: Tokens per mini-batch (default 16)
        ttt_base_lr: Base learning rate for TTT updates
        use_learnable_lr: Enable per-head learnable LR network
        rope_theta: RoPE theta parameter (reserved for future use)
        use_gate: Enable output gating (recommended)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        mini_batch_size: int = 16,
        ttt_base_lr: float = 1.0,
        use_learnable_lr: bool = True,
        rope_theta: float = 10000.0,
        use_gate: bool = True,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_base_lr
        self.use_learnable_lr = use_learnable_lr
        self.use_gate = use_gate
        self.rope_theta = rope_theta

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fast-weight parameters (per-head linear transform)
        # Shape: (num_heads, head_dim, head_dim)
        self.W_init = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)
        self.b_init = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))

        # TTT normalization (per-head, applied to fast-weight output)
        self.ttt_norm = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])

        # Post normalization (applied to final output)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Learnable learning rate network (per-head)
        if use_learnable_lr:
            self.lr_net = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 1, bias=True),
                    nn.Sigmoid()  # Output in [0, 1]
                ) for _ in range(num_heads)
            ])
            # Learnable token index offset
            self.token_idx_offset = nn.Parameter(torch.zeros(mini_batch_size))

        # Fixed token index (1/1, 1/2, 1/3, ..., 1/mini_batch_size)
        self.register_buffer(
            'token_idx',
            1.0 / torch.arange(1, mini_batch_size + 1, dtype=torch.float32)
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Optional gating
        if use_gate:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following official implementation."""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.use_gate:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

    def get_eta(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token, per-head learning rates.

        Official formula (from ttt_layer.py:213-228):
        eta = (base_lr * token_idx) * learnable_lr(X) / head_dim

        Args:
            X: Input (batch, seq_len, hidden_dim)

        Returns:
            eta: Learning rates (batch, num_heads, num_mini_batches, mini_batch_size, 1)
        """
        batch_size, seq_len, _ = X.shape
        num_mini_batches = seq_len // self.mini_batch_size

        if self.use_learnable_lr:
            # Compute learnable LR for each head
            lr_multipliers = []
            for head_idx in range(self.num_heads):
                lr_mult = self.lr_net[head_idx](X)  # (batch, seq_len, 1)
                lr_multipliers.append(lr_mult)
            lr_multipliers = torch.stack(lr_multipliers, dim=1)  # (batch, num_heads, seq_len, 1)

            # Reshape to mini-batches
            lr_multipliers = lr_multipliers.view(
                batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, 1
            )

            # Token index with learnable offset
            token_idx = torch.clamp(self.token_idx + self.token_idx_offset, min=0.0)
        else:
            # Fixed LR
            lr_multipliers = torch.ones(
                batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, 1,
                device=X.device
            )
            token_idx = self.token_idx

        # Compute eta: (base_lr * token_idx) * learnable_lr / head_dim
        eta = (self.ttt_base_lr * token_idx.view(1, 1, 1, -1, 1)) * lr_multipliers / self.head_dim

        return eta

    def process_mini_batch(
        self,
        Q_mb: torch.Tensor,  # (batch, num_heads, mini_batch_size, head_dim)
        K_mb: torch.Tensor,
        V_mb: torch.Tensor,
        eta_mb: torch.Tensor,  # (batch, num_heads, mini_batch_size, 1)
        W: torch.Tensor,  # (num_heads, head_dim, head_dim)
        b: torch.Tensor,  # (num_heads, 1, head_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one mini-batch with analytic solution.

        Official algorithm (from ttt_layer.py:368-429):
        1. Forward: Z = K @ W + b
        2. Apply ttt_norm: ttt_norm_out = ttt_norm(Z)
        3. Compute gradient: grad_normed = ttt_norm_out - (V - K)
        4. Backprop through LayerNorm: grad_Z = vjp(ttt_norm)(grad_normed)
        5. Analytic update with triangular attention:
           Z_bar = Q @ W - (eta * tril(Q @ K^T)) @ grad_Z + b_bar
           where b_bar = b - (eta * tril(ones)) @ grad_Z
        6. Apply ttt_norm and add residual: output = Q + ttt_norm(Z_bar)
        7. Update W, b for next mini-batch:
           W_next = W - (eta_last * K[-1])^T @ grad_Z[-1]
           b_next = b - sum(eta_last * grad_Z[-1])

        Args:
            Q_mb, K_mb, V_mb: Query, Key, Value for this mini-batch
            eta_mb: Learning rates
            W, b: Fast-weight parameters from previous mini-batch

        Returns:
            output_mb: Output for this mini-batch (batch, num_heads, mini_batch_size, head_dim)
            W_next: Updated W for next mini-batch (num_heads, head_dim, head_dim)
            b_next: Updated b for next mini-batch (num_heads, 1, head_dim)
        """
        batch_size, num_heads, mb_size, head_dim = Q_mb.shape

        # Step 1: SSL target: V - K (residual prediction)
        ssl_target = V_mb - K_mb  # (batch, num_heads, mb_size, head_dim)

        # Step 2: Forward pass through fast-weight
        # Z = K @ W + b for each head
        Z = torch.einsum('bhmd,hde->bhme', K_mb, W) + b.unsqueeze(0)  # (batch, num_heads, mb_size, head_dim)

        # Step 3: Apply ttt_norm per head and compute gradient
        Z_normed_list = []
        grad_Z_list = []

        for h in range(num_heads):
            # Apply LayerNorm
            Z_h = Z[:, h]  # (batch, mb_size, head_dim)
            Z_normed_h = self.ttt_norm[h](Z_h)  # (batch, mb_size, head_dim)
            Z_normed_list.append(Z_normed_h)

            # Gradient w.r.t. normalized output
            grad_normed_h = Z_normed_h - ssl_target[:, h]  # (batch, mb_size, head_dim)

            # Backprop through LayerNorm using VJP
            grad_Z_h = layernorm_vjp(
                Z_h,
                grad_normed_h,
                self.ttt_norm[h].weight,
                self.ttt_norm[h].bias,
                eps=self.ttt_norm[h].eps
            )  # (batch, mb_size, head_dim)
            grad_Z_list.append(grad_Z_h)

        grad_Z = torch.stack(grad_Z_list, dim=1)  # (batch, num_heads, mb_size, head_dim)

        # Step 4: Triangular attention: Attn = tril(Q @ K^T)
        Attn = torch.einsum('bhmd,bhnd->bhmn', Q_mb, K_mb)  # (batch, num_heads, mb_size, mb_size)
        Attn = torch.tril(Attn)  # Lower triangular (causal)

        # Step 5: Analytic update for b_bar and Z_bar
        # b_bar = b - (eta * tril(ones)) @ grad_Z
        tril_ones = torch.tril(torch.ones(mb_size, mb_size, device=Q_mb.device))  # (mb_size, mb_size)
        eta_expanded = eta_mb.squeeze(-1).unsqueeze(-1)  # (batch, num_heads, mb_size, 1)

        # Compute b_bar: b - (eta * tril(ones)) @ grad_Z
        # eta: (batch, num_heads, mb_size, 1)
        # tril_ones: (mb_size, mb_size)
        # grad_Z: (batch, num_heads, mb_size, head_dim)
        b_bar = b.unsqueeze(0) - torch.einsum('bhmn,bhnd->bhmd',
                                               eta_mb * tril_ones.unsqueeze(0).unsqueeze(0),
                                               grad_Z)  # (batch, num_heads, 1, head_dim)

        # Z_bar = Q @ W - (eta * Attn) @ grad_Z + b_bar
        Z_bar = torch.einsum('bhmd,hde->bhme', Q_mb, W)  # Q @ W
        Z_bar = Z_bar - torch.einsum('bhmn,bhnd->bhmd', eta_mb * Attn, grad_Z)  # - (eta * Attn) @ grad_Z
        Z_bar = Z_bar + b_bar

        # Step 6: Apply ttt_norm to Z_bar and add residual
        Z_bar_normed_list = []
        for h in range(num_heads):
            Z_bar_normed_h = self.ttt_norm[h](Z_bar[:, h])
            Z_bar_normed_list.append(Z_bar_normed_h)

        Z_bar_normed = torch.stack(Z_bar_normed_list, dim=1)  # (batch, num_heads, mb_size, head_dim)

        # Output: Q + ttt_norm(Z_bar)
        output_mb = Q_mb + Z_bar_normed

        # Step 7: Update W and b for next mini-batch (use last token's gradient)
        eta_last = eta_mb[:, :, -1:, :]  # (batch, num_heads, 1, 1)
        K_last = K_mb[:, :, -1, :]  # (batch, num_heads, head_dim)
        grad_Z_last = grad_Z[:, :, -1, :]  # (batch, num_heads, head_dim)

        # W_next = W - (eta_last * K_last)^T @ grad_Z_last
        # Average over batch (since W is shared across batch)
        # K_last: (batch, num_heads, head_dim) -> bhd
        # (eta_last * grad_Z_last): (batch, num_heads, 1, 1) * (batch, num_heads, head_dim) -> (batch, num_heads, head_dim) -> bhe
        # einsum: bhd,bhe->hde (K^T @ grad), then average over batch
        delta_W = torch.einsum('bhd,bhe->hde',
                                K_last,
                                eta_last.squeeze(-1) * grad_Z_last) / batch_size  # (num_heads, head_dim, head_dim)
        W_next = W - delta_W

        # b_next = b - sum(eta_last * grad_Z_last)
        # Average over batch
        delta_b = (eta_last.squeeze(-1) * grad_Z_last).mean(dim=0)  # (num_heads, head_dim)
        b_next = b - delta_b.unsqueeze(1)  # (num_heads, 1, head_dim)

        return output_mb, W_next, b_next

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with mini-batch analytic TTT.

        Args:
            x: Input (batch, seq_len, hidden_dim)
            return_stats: Return statistics

        Returns:
            output: (batch, seq_len, hidden_dim)
            stats: Optional statistics
        """
        batch_size, seq_len, _ = x.shape

        # Ensure seq_len is divisible by mini_batch_size
        if seq_len % self.mini_batch_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by mini_batch_size ({self.mini_batch_size}). "
                f"Consider padding or using a different mini_batch_size."
            )

        num_mini_batches = seq_len // self.mini_batch_size

        # QKV projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Reshape to mini-batches: (batch, num_heads, num_mini_batches, mini_batch_size, head_dim)
        Q = Q.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)
        K = K.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)
        V = V.view(batch_size, self.num_heads, num_mini_batches, self.mini_batch_size, self.head_dim)

        # Compute learning rates
        eta = self.get_eta(x)  # (batch, num_heads, num_mini_batches, mini_batch_size, 1)

        # Initialize fast-weight
        W = self.W_init.clone()  # (num_heads, head_dim, head_dim)
        b = self.b_init.clone()  # (num_heads, 1, head_dim)

        # Process each mini-batch sequentially
        outputs = []
        for mb_idx in range(num_mini_batches):
            Q_mb = Q[:, :, mb_idx]  # (batch, num_heads, mini_batch_size, head_dim)
            K_mb = K[:, :, mb_idx]
            V_mb = V[:, :, mb_idx]
            eta_mb = eta[:, :, mb_idx]  # (batch, num_heads, mini_batch_size, 1)

            output_mb, W, b = self.process_mini_batch(Q_mb, K_mb, V_mb, eta_mb, W, b)
            outputs.append(output_mb)

        # Concatenate mini-batches
        output = torch.cat(outputs, dim=2)  # (batch, num_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Post normalization (second LayerNorm)
        output = self.post_norm(output)

        # Optional gating
        if self.use_gate:
            gate = self.gate_proj(x)
            gate = F.gelu(gate, approximate='tanh')
            output = gate * output

        # Output projection
        output = self.out_proj(output)

        # Statistics
        stats = None
        if return_stats:
            stats = {
                'num_mini_batches': num_mini_batches,
                'mini_batch_size': self.mini_batch_size,
                'avg_eta': eta.mean().item(),
            }

        return output, stats
