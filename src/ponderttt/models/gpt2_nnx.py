"""
Native GPT-2 implementation using Flax NNX.

This replaces the HuggingFace/Transformers dependency with a pure JAX/Flax implementation,
optimized for TPU deployment.

Based on:
- https://github.com/prabhudavidsheryl/flax_nnx_gpt2
- https://developers.googleblog.com/en/train-gpt2-model-with-jax-on-tpu/
- Official GPT-2 architecture (Radford et al., 2019)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class GPT2Config:
    """GPT-2 model configuration.

    Supports standard GPT-2 variants:
    - gpt2 (124M): 12 layers, 768 dim, 12 heads
    - gpt2-medium (355M): 24 layers, 1024 dim, 16 heads
    - gpt2-large (774M): 36 layers, 1280 dim, 20 heads
    - gpt2-xl (1.5B): 48 layers, 1600 dim, 25 heads
    """
    vocab_size: int = 50257  # GPT-2 BPE vocabulary
    n_positions: int = 1024  # Maximum sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    dropout: float = 0.1  # Dropout probability
    layer_norm_epsilon: float = 1e-5  # Layer norm epsilon

    @classmethod
    def from_pretrained(cls, model_name: str) -> "GPT2Config":
        """Create config for pretrained model."""
        configs = {
            "gpt2": cls(n_layer=12, n_embd=768, n_head=12),  # 124M
            "gpt2-medium": cls(n_layer=24, n_embd=1024, n_head=16),  # 355M
            "gpt2-large": cls(n_layer=36, n_embd=1280, n_head=20),  # 774M
            "gpt2-xl": cls(n_layer=48, n_embd=1600, n_head=25),  # 1.5B
        }
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        return configs[model_name]


class CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPT2Config, rngs: nnx.Rngs):
        """Initialize attention layer.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined QKV projection
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        # Output projection
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
        # Dropout
        self.attn_dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.resid_dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal self-attention.

        Args:
            x: Input tensor [batch, seq_len, n_embd]

        Returns:
            Attention output [batch, seq_len, n_embd]
        """
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Calculate Q, K, V
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # Each [B, T, C]

        # Reshape to [B, n_head, T, head_dim]
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, n_head, T, T]

        # Causal mask: prevent attending to future positions
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn_weights = jnp.where(causal_mask, attn_weights, float('-inf'))

        # Softmax and dropout
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [B, n_head, T, head_dim]

        # Reshape back to [B, T, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection and dropout
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out


class MLP(nnx.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPT2Config, rngs: nnx.Rngs):
        """Initialize MLP.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        self.config = config
        # Expand by 4x (GPT-2 standard)
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply MLP.

        Args:
            x: Input tensor [batch, seq_len, n_embd]

        Returns:
            MLP output [batch, seq_len, n_embd]
        """
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)  # Use approximate GELU like PyTorch
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nnx.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: GPT2Config, rngs: nnx.Rngs):
        """Initialize transformer block.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        self.config = config
        self.ln_1 = nnx.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon, rngs=rngs)
        self.mlp = MLP(config, rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply transformer block.

        Args:
            x: Input tensor [batch, seq_len, n_embd]

        Returns:
            Block output [batch, seq_len, n_embd]
        """
        # Pre-normalization (GPT-2 style)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nnx.Module):
    """GPT-2 language model."""

    def __init__(self, config: GPT2Config, rngs: nnx.Rngs):
        """Initialize GPT-2 model.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        self.config = config

        # Token + position embeddings
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)  # Token
        self.wpe = nnx.Embed(config.n_positions, config.n_embd, rngs=rngs)  # Position
        self.drop = nnx.Dropout(config.dropout, rngs=rngs)

        # Transformer blocks
        self.h = nnx.List([Block(config, rngs) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nnx.LayerNorm(config.n_embd, epsilon=config.layer_norm_epsilon, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Hidden states [batch, seq_len, n_embd]
        """
        B, T = input_ids.shape

        # Get embeddings
        tok_emb = self.wte(input_ids)  # [B, T, n_embd]
        pos = jnp.arange(0, T, dtype=jnp.int32)  # [T]
        pos_emb = self.wpe(pos)  # [T, n_embd]

        # Combine token + position embeddings
        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.h:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        return x


class GPT2LMHeadModel(nnx.Module):
    """GPT-2 with language modeling head."""

    def __init__(
        self,
        config: GPT2Config,
        rngs: nnx.Rngs,
        tie_word_embeddings: bool = True
    ):
        """Initialize GPT-2 LM model.

        Args:
            config: Model configuration
            rngs: Random number generators
            tie_word_embeddings: If True, share embedding weights with LM head
        """
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings

        # Transformer
        self.transformer = GPT2Model(config, rngs)

        # LM head (only used if not tying weights)
        if not tie_word_embeddings:
            self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Forward pass with language modeling head.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Get hidden states from transformer
        hidden_states = self.transformer(input_ids)

        # Compute logits
        if self.tie_word_embeddings:
            # Share weights with token embedding (reduces parameters by 31%)
            # Use transpose: embedding is [vocab_size, n_embd], need [n_embd, vocab_size]
            embedding_kernel = self.transformer.wte.embedding.value
            logits = hidden_states @ embedding_kernel.T
        else:
            logits = self.lm_head(hidden_states)

        return logits


def load_gpt2_model(
    model_name: str = "gpt2",
    tie_word_embeddings: bool = True,
    seed: int = 0
) -> tuple[GPT2LMHeadModel, GPT2Config]:
    """Create GPT-2 model with random initialization.

    Args:
        model_name: Model variant (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
        tie_word_embeddings: If True, share embedding weights with LM head
        seed: Random seed

    Returns:
        (model, config) tuple
    """
    config = GPT2Config.from_pretrained(model_name)
    rngs = nnx.Rngs(seed)
    model = GPT2LMHeadModel(config, rngs, tie_word_embeddings=tie_word_embeddings)
    return model, config
