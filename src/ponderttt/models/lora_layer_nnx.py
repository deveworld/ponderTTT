"""
LoRA (Low-Rank Adaptation) layer in NNX.

Alternative to TTT Layer for fast-weight adaptation.
Provides similar interface for easy switching.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class LoRAConfig:
    """Configuration for LoRA layer."""

    hidden_dim: int = 768
    rank: int = 64  # Low-rank dimension (64/128/256)
    alpha: float = 64.0  # Scaling factor (usually = rank)
    dropout_rate: float = 0.1
    target_modules: tuple = ("q_proj", "v_proj")  # Which attention modules to adapt
    dtype: jnp.dtype = jnp.float32
    initializer_range: float = 0.02


class LoRALinear(nnx.Module):
    """
    LoRA adaptation for a single linear layer.

    Implements: h_out = W @ h + (B @ A @ h) * (alpha / rank)
    where W is frozen, A and B are trainable low-rank matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ):
        """Initialize LoRA linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Low-rank dimension
            alpha: Scaling factor
            dropout_rate: Dropout probability
            rngs: Random number generators
        """
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices (A: down-projection, B: up-projection)
        # A initialized with Kaiming uniform, B initialized to zero
        self.lora_A = nnx.Linear(
            in_features,
            rank,
            use_bias=False,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )

        self.lora_B = nnx.Linear(
            rank,
            out_features,
            use_bias=False,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # Dropout for LoRA path
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor [batch, seq_len, in_features]
            train: Whether in training mode (for dropout)

        Returns:
            LoRA output [batch, seq_len, out_features]
        """
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_out = self.lora_A(x)
        lora_out = self.dropout(lora_out, deterministic=not train)
        lora_out = self.lora_B(lora_out)

        # Scale by alpha/rank
        return lora_out * self.scaling


class LoRALayer(nnx.Module):
    """
    LoRA adaptation layer for attention mechanism.

    Adapts Q and V projections in multi-head attention.
    Provides same interface as TTTLayer for easy switching.
    """

    def __init__(self, config: LoRAConfig, rngs: nnx.Rngs):
        """Initialize LoRA layer.

        Args:
            config: LoRA configuration
            rngs: Random number generators
        """
        self.config = config

        # LoRA adapters for Q and V projections
        # Following LaCT: only adapt query and value
        self.q_lora = LoRALinear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )

        self.v_lora = LoRALinear(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: Optional[jax.Array] = None,
        position_ids: Optional[jax.Array] = None,
        *,
        train: bool = False,
    ) -> Tuple[jax.Array, dict]:
        """Forward pass with LoRA adaptation.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            mask: Attention mask (unused, for interface compatibility)
            position_ids: Position IDs (unused, for interface compatibility)
            train: Whether in training mode

        Returns:
            Tuple of:
                - adapted_hidden: Adapted hidden states [batch, seq_len, hidden_dim]
                - stats: Dictionary with adaptation statistics
        """
        # Apply LoRA to Q and V projections
        q_delta = self.q_lora(hidden_states, train=train)  # [batch, seq, hidden]
        v_delta = self.v_lora(hidden_states, train=train)  # [batch, seq, hidden]

        # Simple self-attention using adapted Q/V (single-head)
        q = hidden_states + q_delta
        k = hidden_states
        v = hidden_states + v_delta

        scale = 1.0 / jnp.sqrt(self.config.hidden_dim)
        attn_scores = jnp.einsum("bth,bsh->bts", q, k) * scale
        if mask is not None:
            attn_mask = (mask[:, None, :] > 0).astype(jnp.float32)
            attn_scores = jnp.where(attn_mask > 0, attn_scores, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        adapted_output = attn_weights @ v

        # Compute statistics (for monitoring)
        stats = {
            "q_norm": jnp.linalg.norm(q_delta),
            "v_norm": jnp.linalg.norm(v_delta),
            "q_mean": jnp.mean(jnp.abs(q_delta)),
            "v_mean": jnp.mean(jnp.abs(v_delta)),
            "output_norm": jnp.linalg.norm(adapted_output),
            "attn_entropy": -jnp.mean(attn_weights * jnp.log(attn_weights + 1e-10)),
        }

        return adapted_output, stats

    def reset_parameters(self):
        """Reset LoRA parameters (for test-time training)."""
        # Re-initialize B matrices to zero (A keeps random init)
        self.q_lora.lora_B.kernel.value = jnp.zeros_like(
            self.q_lora.lora_B.kernel.value
        )
        self.v_lora.lora_B.kernel.value = jnp.zeros_like(
            self.v_lora.lora_B.kernel.value
        )


def count_lora_parameters(config: LoRAConfig) -> int:
    """Count number of trainable parameters in LoRA layer.

    Args:
        config: LoRA configuration

    Returns:
        Number of parameters
    """
    # Q LoRA: (hidden x rank) + (rank x hidden)
    q_params = 2 * config.hidden_dim * config.rank

    # V LoRA: (hidden x rank) + (rank x hidden)
    v_params = 2 * config.hidden_dim * config.rank

    total = q_params + v_params

    return total


def compare_parameters():
    """Compare parameter counts: LoRA vs TTT Layer vs Full Attention.

    For GPT-2 (hidden_dim=768):
    - Full attention: 768^2 x 3 (Q,K,V) = 1,769,472
    - TTT Layer: ~2M parameters (mini-batch GD)
    - LoRA (r=64): 2 x 768 x 64 = 98,304 (94% reduction)
    - LoRA (r=128): 2 x 768 x 128 = 196,608 (89% reduction)
    """
    hidden_dim = 768

    print("Parameter comparison for GPT-2 (hidden_dim=768):")
    print(f"{'Method':<20} {'Parameters':>12} {'Reduction':>10}")
    print("-" * 45)

    # Full attention
    full = hidden_dim * hidden_dim * 3
    print(f"{'Full Attention':<20} {full:>12,} {'-':>10}")

    # LoRA variants
    for rank in [64, 128, 256]:
        config = LoRAConfig(hidden_dim=hidden_dim, rank=rank)
        lora_params = count_lora_parameters(config)
        reduction = (1 - lora_params / full) * 100
        print(f"{'LoRA (r=' + str(rank) + ')':<20} {lora_params:>12,} {reduction:>9.1f}%")


if __name__ == "__main__":
    # Test LoRA layer
    print("Testing LoRA Layer...")

    config = LoRAConfig(hidden_dim=768, rank=64)
    rngs = nnx.Rngs(0)
    lora = LoRALayer(config, rngs)

    # Test forward pass
    batch_size, seq_len = 2, 512
    hidden_states = jnp.ones((batch_size, seq_len, config.hidden_dim))

    output, stats = lora(hidden_states, train=True)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Stats: {stats}")

    # Parameter count
    print("\n" + "=" * 45)
    compare_parameters()
