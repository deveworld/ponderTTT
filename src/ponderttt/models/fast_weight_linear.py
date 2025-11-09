"""
Linear Fast-Weight Module (Official TTT style).

Simplified version matching the official TTT implementation:
- Single linear transformation: W @ x + b
- Optional layer normalization
- No hidden MLP layers

This is more efficient and matches the official implementation.
"""

from typing import Optional

import torch
import torch.nn as nn


class LinearFastWeightModule(nn.Module):
    """
    Linear fast-weight transformation (official TTT style).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        use_layer_norm: Whether to apply layer normalization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Single linear layer
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (..., input_dim)

        Returns:
            output: (..., output_dim)
        """
        out = self.linear(x)

        if self.use_layer_norm:
            out = self.ln(out)

        return out

    def clone(self) -> 'LinearFastWeightModule':
        """Create a clone with copied parameters."""
        cloned = LinearFastWeightModule(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_layer_norm=self.use_layer_norm,
        )

        # Copy parameters
        cloned.load_state_dict(self.state_dict())

        return cloned


class MultiHeadLinearFastWeight(nn.Module):
    """
    Multi-head linear fast-weight module.

    Each head has its own linear transformation.

    Args:
        num_heads: Number of heads
        input_dim: Input dimension per head
        output_dim: Output dimension per head
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create one module per head
        self.heads = nn.ModuleList([
            LinearFastWeightModule(
                input_dim=input_dim,
                output_dim=output_dim,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, num_heads, seq_len, input_dim)

        Returns:
            output: (batch, num_heads, seq_len, output_dim)
        """
        batch_size, num_heads, seq_len, _ = x.shape
        assert num_heads == self.num_heads

        outputs = []
        for h in range(num_heads):
            out_h = self.heads[h](x[:, h, :, :])  # (batch, seq_len, output_dim)
            outputs.append(out_h)

        output = torch.stack(outputs, dim=1)  # (batch, num_heads, seq_len, output_dim)

        return output

    def clone(self) -> 'MultiHeadLinearFastWeight':
        """Create a clone with copied parameters."""
        cloned = MultiHeadLinearFastWeight(
            num_heads=self.num_heads,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_layer_norm=self.heads[0].use_layer_norm,
        )

        # Copy parameters
        cloned.load_state_dict(self.state_dict())

        return cloned
