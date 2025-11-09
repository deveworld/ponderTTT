"""
Iterative TTT Layer V2 - Configurable Variants.

Supports multiple configurations for ablation studies:
- Fast-weight type: 'linear' (official-like) or 'mlp' (enhanced)
- TTT loss: with/without projection layer
- Learning rate: fixed or position-dependent
- Processing: sequential (current) or bucketed (future optimization)

This allows systematic comparison of design choices.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_weight import MultiHeadFastWeight
from .fast_weight_linear import MultiHeadLinearFastWeight


class IterativeTTTLayerV2(nn.Module):
    """
    Configurable iterative TTT layer.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        fast_weight_type: 'linear' or 'mlp'
        fast_weight_hidden_dim: Hidden dim for MLP variant (ignored if linear)
        base_lr: Base learning rate
        max_steps: Maximum gradient steps
        ttt_loss_type: 'reconstruction' or 'prediction'
        ttt_loss_projection: Whether to use additional projection layer
        use_sequential: Carry forward fast-weight states
        use_layer_norm: Layer normalization in fast weights
        lr_schedule: 'fixed' or 'position_dependent'
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        fast_weight_type: str = 'linear',  # 'linear' or 'mlp'
        fast_weight_hidden_dim: int = 64,
        base_lr: float = 0.1,
        max_steps: int = 8,
        ttt_loss_type: str = "reconstruction",
        ttt_loss_projection: bool = False,  # Official: False
        use_sequential: bool = True,
        use_layer_norm: bool = True,
        lr_schedule: str = 'fixed',  # 'fixed' or 'position_dependent'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        assert hidden_dim == num_heads * self.head_dim

        self.fast_weight_type = fast_weight_type
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.ttt_loss_type = ttt_loss_type
        self.ttt_loss_projection = ttt_loss_projection
        self.use_sequential = use_sequential
        self.lr_schedule = lr_schedule

        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fast-weight module - configurable
        if fast_weight_type == 'linear':
            self.fast_weight = MultiHeadLinearFastWeight(
                num_heads=num_heads,
                input_dim=self.head_dim,
                output_dim=self.head_dim,
                use_layer_norm=use_layer_norm,
            )
        elif fast_weight_type == 'mlp':
            self.fast_weight = MultiHeadFastWeight(
                num_heads=num_heads,
                input_dim=self.head_dim,
                hidden_dim=fast_weight_hidden_dim,
                output_dim=self.head_dim,
                use_layer_norm=use_layer_norm,
            )
        else:
            raise ValueError(f"Unknown fast_weight_type: {fast_weight_type}")

        # Optional TTT loss projection
        if ttt_loss_projection:
            self.ttt_loss_head = nn.Linear(self.head_dim, self.head_dim, bias=True)
        else:
            self.ttt_loss_head = None

        # Position-dependent LR (if enabled)
        if lr_schedule == 'position_dependent':
            self.register_buffer(
                'position_scale',
                torch.arange(1, max_steps + 1, dtype=torch.float32) / max_steps
            )
        else:
            self.position_scale = None

        # TTT normalization (per-head, applied to fast-weight output BEFORE residual)
        # Official implementation applies LayerNorm to z_t before adding to q_t
        self.ttt_norm = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads)
        ])

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Post normalization (applied to final output AFTER residual)
        # Official implementation applies this after Q + ttt_norm(Z)
        self.post_norm = nn.LayerNorm(hidden_dim)

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
        if self.ttt_loss_head is not None:
            nn.init.normal_(self.ttt_loss_head.weight, std=0.02)
            nn.init.zeros_(self.ttt_loss_head.bias)
        if self.use_gate:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

    def _get_lr(self, step: int, position: int) -> float:
        """
        Get learning rate for this step.

        Args:
            step: Current gradient step (0-indexed)
            position: Token position in sequence

        Returns:
            lr: Learning rate
        """
        if self.lr_schedule == 'position_dependent':
            # Scale by position and step
            position_factor = (position + 1) / self.max_steps
            step_factor = self.position_scale[min(step, len(self.position_scale) - 1)]
            return self.base_lr * position_factor * step_factor / self.head_dim
        else:
            # Fixed LR
            return self.base_lr

    def _compute_ttt_loss(
        self,
        fast_weight,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TTT self-supervised loss (simplified version).

        Args:
            fast_weight: Current fast-weight module
            k_t: Key (batch, num_heads, head_dim)
            v_t: Value (batch, num_heads, head_dim)

        Returns:
            loss: Scalar loss
        """
        # Forward through fast-weight
        k_t_expanded = k_t.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
        z_t = fast_weight(k_t_expanded).squeeze(2)  # (batch, num_heads, head_dim)

        # Target: V - K (residual)
        target = v_t - k_t

        # Optional projection
        if self.ttt_loss_head is not None:
            predictions = []
            for h in range(self.num_heads):
                pred_h = self.ttt_loss_head(z_t[:, h, :])
                predictions.append(pred_h)
            pred = torch.stack(predictions, dim=1)
        else:
            # Direct prediction (official style)
            pred = z_t

        # MSE loss
        loss = F.mse_loss(pred, target, reduction='mean')

        return loss

    def _update_fast_weight(
        self,
        fast_weight,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        lr: float,
        create_graph: bool = True,
    ):
        """
        Perform one gradient step.

        Args:
            fast_weight: Current fast-weight module
            k_t: Key tensor
            v_t: Value tensor
            lr: Learning rate
            create_graph: Enable meta-learning

        Returns:
            updated_fast_weight: Updated module
        """
        loss = self._compute_ttt_loss(fast_weight, k_t, v_t)

        grads = torch.autograd.grad(
            loss,
            fast_weight.parameters(),
            create_graph=create_graph,
            retain_graph=True,
        )

        updated_fast_weight = fast_weight.clone()

        with torch.no_grad():
            for param, grad in zip(updated_fast_weight.parameters(), grads):
                param.data = param.data - lr * grad

        return updated_fast_weight

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[torch.Tensor] = None,
        prev_fast_weight = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[nn.Module]]:
        """
        Forward pass with iterative gradient descent.

        Args:
            x: Input (batch, seq_len, hidden_dim)
            num_steps: Steps per token (seq_len,) or (batch, seq_len)
            prev_fast_weight: Previous fast-weight state
            return_stats: Return statistics

        Returns:
            output: (batch, seq_len, hidden_dim)
            stats: Optional statistics
            final_fast_weight: Final fast-weight state
        """
        batch_size, seq_len, _ = x.shape

        # Normalize num_steps
        if num_steps is None:
            num_steps = torch.full(
                (seq_len,), self.max_steps,
                dtype=torch.float32, device=x.device
            )
        elif isinstance(num_steps, int):
            num_steps = torch.full(
                (seq_len,), num_steps,
                dtype=torch.float32, device=x.device
            )
        elif num_steps.ndim == 0:
            num_steps = torch.full(
                (seq_len,), num_steps.item(),
                dtype=torch.float32, device=x.device
            )
        elif num_steps.ndim == 1:
            num_steps = num_steps.float().to(x.device)
        elif num_steps.ndim == 2:
            num_steps = num_steps.float().to(x.device)
        else:
            raise ValueError(f"Invalid num_steps shape: {num_steps.shape}")

        # QKV projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Initialize fast-weight
        if self.use_sequential and prev_fast_weight is not None:
            current_fast_weight = prev_fast_weight
        else:
            current_fast_weight = self.fast_weight

        # Determine if per-token
        per_token = (num_steps.ndim == 2)

        # Process each token
        outputs = []
        total_steps_taken = []

        for t in range(seq_len):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            if per_token:
                steps_t = num_steps[:, t].max()
                steps_t_int = int(steps_t.item())
            else:
                steps_t = num_steps[t]
                steps_t_int = int(steps_t.item())

            if self.use_sequential:
                fast_weight_t = current_fast_weight
            else:
                fast_weight_t = current_fast_weight.clone()

            # K-step gradient descent
            for step in range(steps_t_int):
                lr = self._get_lr(step, t)
                fast_weight_t = self._update_fast_weight(
                    fast_weight_t, k_t, v_t,
                    lr=lr,
                    create_graph=self.training,
                )

            # Compute output
            k_t_expanded = k_t.unsqueeze(2)
            z_t = fast_weight_t(k_t_expanded).squeeze(2)

            # Apply per-head TTT normalization BEFORE residual (Official TTT-Linear style)
            z_t_normed_list = []
            for h in range(self.num_heads):
                z_t_h = z_t[:, h, :]  # (batch, head_dim)
                z_t_normed_h = self.ttt_norm[h](z_t_h)  # (batch, head_dim)
                z_t_normed_list.append(z_t_normed_h)
            z_t_normed = torch.stack(z_t_normed_list, dim=1)  # (batch, num_heads, head_dim)

            # Add to query (like in original TTT): output = Q + ttt_norm(Z)
            output_t = q_t + z_t_normed

            outputs.append(output_t)
            total_steps_taken.append(steps_t)

        # Stack outputs
        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Post normalization AFTER residual (second LayerNorm)
        # Official implementation: Z = post_norm(Q + ttt_norm(Z))
        output = self.post_norm(output)

        # Optional gating
        if self.use_gate:
            gate = self.gate_proj(x)
            gate = F.gelu(gate, approximate="tanh")
            output = gate * output

        # Output projection
        output = self.out_proj(output)

        # Statistics
        stats = None
        if return_stats:
            steps_taken = torch.stack(total_steps_taken, dim=0)
            stats = {
                'avg_steps': steps_taken.mean().item(),
                'min_steps': steps_taken.min().item(),
                'max_steps': steps_taken.max().item(),
                'fast_weight_type': self.fast_weight_type,
                'ttt_loss_projection': self.ttt_loss_projection,
                'lr_schedule': self.lr_schedule,
            }

        final_fast_weight = current_fast_weight if self.use_sequential else None

        return output, stats, final_fast_weight
