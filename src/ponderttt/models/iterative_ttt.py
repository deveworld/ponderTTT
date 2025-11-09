"""
Iterative TTT Layer with explicit gradient descent loops.

Implements per-token adaptive iteration counts via explicit K-step gradient descent.
This is the core innovation of PonderTTT: different tokens receive different numbers
of gradient updates based on difficulty.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_weight import FastWeightModule, MultiHeadFastWeight


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


class IterativeTTTLayer(nn.Module):
    """
    Iterative TTT layer with explicit gradient descent.

    Unlike the analytic TTT layer which uses a closed-form triangular solve,
    this layer performs explicit gradient descent steps. The number of steps
    can vary per token, enabling true adaptive computation.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (hidden_dim // num_heads)
        fast_weight_hidden_dim: Hidden dimension for fast-weight MLP
        base_lr: Base learning rate for gradient descent
        max_steps: Maximum number of gradient steps per token
        ttt_loss_type: Type of TTT loss ('reconstruction' or 'prediction')
        use_sequential: If True, carry forward fast weights across tokens
        use_layer_norm: Whether to use layer normalization in fast weights
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        fast_weight_hidden_dim: int = 64,
        base_lr: float = 0.1,
        max_steps: int = 8,
        ttt_loss_type: str = "reconstruction",
        use_sequential: bool = True,
        use_layer_norm: bool = True,
        use_learnable_lr: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        assert hidden_dim == num_heads * self.head_dim, "hidden_dim must be divisible by num_heads"

        self.fast_weight_hidden_dim = fast_weight_hidden_dim
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.ttt_loss_type = ttt_loss_type
        self.use_sequential = use_sequential
        self.use_learnable_lr = use_learnable_lr

        # QKV projections (similar to standard attention)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fast-weight module (replaces W1/b1 in analytic TTT)
        # Each head has its own fast-weight module
        self.fast_weight = MultiHeadFastWeight(
            num_heads=num_heads,
            input_dim=self.head_dim,
            hidden_dim=fast_weight_hidden_dim,
            output_dim=self.head_dim,
            use_layer_norm=use_layer_norm,
        )

        # TTT loss head (for self-supervised learning)
        if ttt_loss_type == "reconstruction":
            # Simple reconstruction: predict input from fast-weight output
            self.ttt_loss_head = nn.Linear(self.head_dim, self.head_dim, bias=True)
        elif ttt_loss_type == "prediction":
            # Predict next token representation
            self.ttt_loss_head = nn.Linear(self.head_dim, self.head_dim, bias=True)
        else:
            raise ValueError(f"Unknown ttt_loss_type: {ttt_loss_type}")

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

        # Learnable LR modulation (optional)
        if use_learnable_lr:
            # Output dimension is 1 (scalar multiplier per token)
            self.lr_network = LearnableLRNetwork(hidden_dim, output_dim=1)
        else:
            self.lr_network = None

        # Optional gating (like in original TTT)
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
        nn.init.normal_(self.ttt_loss_head.weight, std=0.02)
        nn.init.zeros_(self.ttt_loss_head.bias)
        if self.use_gate:
            nn.init.normal_(self.gate_proj.weight, std=0.02)

    def _compute_ttt_loss(
        self,
        fast_weight: MultiHeadFastWeight,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TTT self-supervised loss.

        Args:
            fast_weight: Current fast-weight module
            k_t: Key for current token (batch, num_heads, head_dim)
            v_t: Value for current token (batch, num_heads, head_dim)

        Returns:
            loss: Scalar loss value for this token
        """
        batch_size = k_t.shape[0]

        # Pass key through fast-weight module
        # k_t: (batch, num_heads, head_dim)
        k_t_expanded = k_t.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
        z_t = fast_weight(k_t_expanded).squeeze(2)  # (batch, num_heads, head_dim)

        # Compute reconstruction target
        if self.ttt_loss_type == "reconstruction":
            # Target: reconstruct V - K (residual)
            target = v_t - k_t
        else:
            # For prediction, target would be next token's value
            # For now, use same as reconstruction
            target = v_t - k_t

        # Predict target from z_t
        predictions = []
        for h in range(self.num_heads):
            pred_h = self.ttt_loss_head(z_t[:, h, :])  # (batch, head_dim)
            predictions.append(pred_h)
        pred = torch.stack(predictions, dim=1)  # (batch, num_heads, head_dim)

        # MSE loss
        loss = F.mse_loss(pred, target, reduction='mean')

        return loss

    def _update_fast_weight(
        self,
        fast_weight: MultiHeadFastWeight,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        lr: float,
        create_graph: bool = True,
    ) -> MultiHeadFastWeight:
        """
        Perform one gradient step on fast-weight module.

        Args:
            fast_weight: Current fast-weight module
            k_t: Key for current token
            v_t: Value for current token
            lr: Learning rate for this step
            create_graph: Whether to create computation graph (for meta-learning)

        Returns:
            updated_fast_weight: Updated fast-weight module
        """
        loss = self._compute_ttt_loss(fast_weight, k_t, v_t)

        grads = torch.autograd.grad(
            loss,
            fast_weight.parameters(),
            create_graph=create_graph,
            retain_graph=True,
        )

        updated_fast_weight = fast_weight.clone()

        # Update parameters
        # NOTE: Currently uses no_grad() which prevents gradient flow to learnable LR
        # Full learnable LR support would require architectural changes to avoid
        # in-place operations on leaf variables
        with torch.no_grad():
            for param, grad in zip(updated_fast_weight.parameters(), grads):
                if isinstance(lr, torch.Tensor):
                    param.data = param.data - lr.item() * grad
                else:
                    param.data = param.data - lr * grad

        return updated_fast_weight

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[torch.Tensor] = None,
        prev_fast_weight: Optional[MultiHeadFastWeight] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[MultiHeadFastWeight]]:
        """
        Forward pass with iterative gradient descent.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            num_steps: Number of gradient steps per token
                      - None: uses max_steps for all tokens
                      - int: fixed steps for all tokens
                      - (seq_len,): per-position allocation
                      - (batch, seq_len): true per-token allocation
            prev_fast_weight: Previous fast-weight state (for sequential mode)
            return_stats: Whether to return detailed statistics

        Returns:
            output: Output tensor (batch, seq_len, hidden_dim)
            stats: Optional statistics dictionary
            final_fast_weight: Final fast-weight state (for sequential mode)
        """
        batch_size, seq_len, _ = x.shape

        # Normalize num_steps to (batch, seq_len) or (seq_len,)
        if num_steps is None:
            num_steps = torch.full(
                (seq_len,),
                self.max_steps,
                dtype=torch.float32,
                device=x.device
            )
        elif isinstance(num_steps, int):
            num_steps = torch.full(
                (seq_len,),
                num_steps,
                dtype=torch.float32,
                device=x.device
            )
        elif num_steps.ndim == 0:
            num_steps = torch.full(
                (seq_len,),
                num_steps.item(),
                dtype=torch.float32,
                device=x.device
            )
        elif num_steps.ndim == 1:
            # Per-position: (seq_len,)
            num_steps = num_steps.float().to(x.device)
        elif num_steps.ndim == 2:
            # Per-token: (batch, seq_len)
            num_steps = num_steps.float().to(x.device)
        else:
            raise ValueError(f"num_steps must be scalar, 1D, or 2D, got shape {num_steps.shape}")

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Initialize fast-weight state
        if self.use_sequential and prev_fast_weight is not None:
            current_fast_weight = prev_fast_weight
        else:
            current_fast_weight = self.fast_weight

        # Determine if per-token or per-position
        per_token = (num_steps.ndim == 2)

        # Process each token sequentially
        outputs = []
        total_steps_taken = []

        for t in range(seq_len):
            q_t = q[:, :, t, :]  # (batch, num_heads, head_dim)
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            if per_token:
                # True per-token: different steps for each batch element
                # Use max steps across batch for this position (conservative)
                steps_t = num_steps[:, t].max()
                steps_t_int = int(steps_t.item())
            else:
                # Per-position: same steps for all batch elements
                steps_t = num_steps[t]
                steps_t_int = int(steps_t.item())

            if self.use_sequential:
                fast_weight_t = current_fast_weight
            else:
                fast_weight_t = current_fast_weight.clone()

            # Compute learning rate (with optional learned modulation)
            if self.use_learnable_lr:
                # Get current token's hidden state
                x_t = x[:, t, :]  # (batch, hidden_dim)
                # Compute LR multiplier (batch, 1)
                lr_mult = self.lr_network(x_t)  # (batch, 1)
                # Use mean across batch for stability (keep as tensor for gradient)
                lr_mult_mean = lr_mult.mean()  # scalar tensor
                # Clamp to reasonable range
                lr_mult_mean = torch.clamp(lr_mult_mean, 0.1, 2.0)
                lr_t = self.base_lr * lr_mult_mean  # scalar tensor
            else:
                lr_t = self.base_lr

            for step in range(steps_t_int):
                fast_weight_t = self._update_fast_weight(
                    fast_weight_t,
                    k_t,
                    v_t,
                    lr=lr_t,
                    create_graph=self.training,
                )

            # Use final fast-weight state to compute output
            k_t_expanded = k_t.unsqueeze(2)  # (batch, num_heads, 1, head_dim)
            z_t = fast_weight_t(k_t_expanded).squeeze(2)  # (batch, num_heads, head_dim)

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

            # Sequential mode: state already updated in-place
            # Independent mode: discard fast_weight_t after use

        # Stack outputs: (batch, num_heads, seq_len, head_dim)
        output = torch.stack(outputs, dim=2)

        # Reshape back to (batch, seq_len, hidden_dim)
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
                "avg_steps": steps_taken.mean().item(),
                "min_steps": steps_taken.min().item(),
                "max_steps": steps_taken.max().item(),
                "steps_distribution": steps_taken.cpu().tolist(),
            }

        final_fast_weight = current_fast_weight if self.use_sequential else None

        return output, stats, final_fast_weight

    def count_flops(self, seq_len: int, num_steps: torch.Tensor) -> int:
        """
        Count FLOPs for this layer given specific step allocations.

        Args:
            seq_len: Sequence length
            num_steps: Number of steps per token (seq_len,)

        Returns:
            total_flops: Total FLOPs for this forward pass
        """
        # QKV projections: 3 * (2 * hidden_dim * hidden_dim) per token
        qkv_flops = 3 * 2 * self.hidden_dim * self.hidden_dim * seq_len

        # For each token and each step:
        # - Fast-weight forward: 2 * (head_dim * fast_weight_hidden_dim + fast_weight_hidden_dim * head_dim) per head
        # - Loss computation: head_dim ops per head (MSE)
        # - Backward pass: 2x forward (standard autograd)
        # - Parameter update: negligible
        # Total per step: forward + backward = 3x forward

        fast_weight_forward_flops_per_head = (
            2 * self.head_dim * self.fast_weight_hidden_dim +  # fc1
            2 * self.fast_weight_hidden_dim * self.head_dim    # fc2
        )
        step_flops_per_token = (
            self.num_heads * fast_weight_forward_flops_per_head * 3  # 1x forward + 2x backward
        )

        # Sum over all tokens and their steps
        total_step_flops = (num_steps.sum() * step_flops_per_token).item()

        # Output projection: 2 * hidden_dim * hidden_dim per token
        out_proj_flops = 2 * self.hidden_dim * self.hidden_dim * seq_len

        total_flops = qkv_flops + total_step_flops + out_proj_flops

        return int(total_flops)
