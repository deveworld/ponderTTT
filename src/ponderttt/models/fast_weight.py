"""
Fast-weight module for iterative TTT.

Implements a small MLP that serves as the fast-adapting memory for test-time training.
This module is updated via gradient descent during inference for each token.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastWeightModule(nn.Module):
    """
    Fast-weight module: a small MLP that adapts during test-time.

    This module replaces the linear W1/b1 parameters in the original TTT layer.
    Instead of a simple linear transformation, it uses a small neural network
    that can be updated via gradient descent for each token.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer (typically small, e.g., 64-128)
        output_dim: Dimension of output features
        use_layer_norm: Whether to use layer normalization
        activation: Activation function ('gelu' or 'relu')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Two-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Optional layer norm for stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)

        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fast-weight module.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            output: Output tensor of shape (..., output_dim)
        """
        h = self.fc1(x)
        if self.use_layer_norm:
            h = self.ln(h)
        h = self.activation(h)
        output = self.fc2(h)
        return output

    def clone(self) -> "FastWeightModule":
        """
        Create a deep copy of this module with the same parameters.

        This is used to create per-token fast-weight instances that can be
        independently updated during the iterative gradient descent process.

        Returns:
            A new FastWeightModule with copied parameters
        """
        new_module = FastWeightModule(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            use_layer_norm=self.use_layer_norm,
        )
        new_module.load_state_dict(self.state_dict())
        return new_module

    def get_flattened_params(self) -> torch.Tensor:
        """Get all parameters as a single flattened vector."""
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_flattened_params(self, flat_params: torch.Tensor):
        """Set all parameters from a flattened vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = flat_params[offset:offset + numel].view(p.shape)
            offset += numel


class MultiHeadFastWeight(nn.Module):
    """
    Multi-head version of FastWeightModule.

    Maintains separate fast-weight modules for each attention head,
    similar to how multi-head attention works.

    Args:
        num_heads: Number of attention heads
        input_dim: Dimension of input per head
        hidden_dim: Hidden dimension per head
        output_dim: Output dimension per head
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Create separate fast-weight module for each head
        self.heads = nn.ModuleList([
            FastWeightModule(input_dim, hidden_dim, output_dim, use_layer_norm)
            for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all heads.

        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, input_dim)

        Returns:
            output: Output tensor of shape (batch, num_heads, seq_len, output_dim)
        """
        batch_size, num_heads, seq_len, _ = x.shape
        assert num_heads == self.num_heads

        # Process each head separately
        outputs = []
        for h in range(num_heads):
            head_input = x[:, h, :, :]  # (batch, seq_len, input_dim)
            head_output = self.heads[h](head_input)
            outputs.append(head_output)

        # Stack outputs: (batch, num_heads, seq_len, output_dim)
        output = torch.stack(outputs, dim=1)
        return output

    def clone(self) -> "MultiHeadFastWeight":
        """Create a deep copy with the same parameters."""
        new_module = MultiHeadFastWeight(
            num_heads=self.num_heads,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )
        new_module.load_state_dict(self.state_dict())
        return new_module

    def get_head(self, head_idx: int) -> FastWeightModule:
        """Get a specific head's fast-weight module."""
        return self.heads[head_idx]
