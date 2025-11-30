"""Gemma 3 model with Test-Time Training (TTT) layer integration.

Combines the official Flax Gemma architecture with PonderTTT's TTT layer
for adaptive compute allocation during inference.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    Array: TypeAlias = jax.Array
    Cache: TypeAlias = dict[str, dict[str, jax.Array]]
else:
    Array = jax.Array
    Cache = dict

from .config import Gemma3Config
from .layers import RMSNorm
from .modules import Block, Embedder
from .sow_lib import SowConfig
from ..ttt_layer_nnx import TTTLayer, TTTConfig


def maybe_with_partitioning(fn, axis_rules, axis_rules_args=()):
    """Apply partitioning if axis_rules are provided."""
    if axis_rules is None:
        return fn
    return nnx.with_partitioning(fn, axis_rules(*axis_rules_args))


class Gemma3Model(nnx.Module):
    """Gemma 3 base model (without LM head).

    This is the core transformer model that produces hidden states.
    """

    def __init__(
        self,
        config: Gemma3Config,
        *,
        rngs: nnx.Rngs,
        sow_config: SowConfig = SowConfig(),
    ):
        self.config = config
        self.sow_config = sow_config

        # Token embeddings
        self.embedder = Embedder(
            vocab_size=config.num_embed,
            embed_dim=config.embed_dim,
            embedding_init=maybe_with_partitioning(
                nnx.initializers.normal(),
                config.axis_rules,
                ("vocab", "embed"),
            ),
            dtype=config.dtype,
            rngs=rngs,
        )

        # Transformer layers
        self.layers = nnx.List([
            Block(
                config=config,
                attn_type=attn_type,
                sow_config=sow_config,
                rngs=rngs,
            )
            for attn_type in config.attention_types
        ])

        # Final layer norm
        self.final_norm = RMSNorm(
            config.embed_dim,
            scale_init=maybe_with_partitioning(
                nnx.initializers.zeros_init(),
                config.axis_rules,
                ("embed",),
            ),
            rngs=rngs,
            dtype=config.dtype,
        )

    def __call__(
        self,
        tokens: Array,
        positions: Array,
        cache: Cache | None = None,
        attention_mask: Array | None = None,
    ) -> tuple[Array, Cache | None]:
        """Forward pass through the model.

        Args:
            tokens: Input token IDs [batch, seq_len]
            positions: Position IDs [batch, seq_len]
            cache: KV cache for incremental decoding
            attention_mask: Attention mask [batch, seq_len, seq_len]

        Returns:
            Tuple of (hidden_states, new_cache)
        """
        new_cache = None if cache is None else {}

        # Embed tokens
        x = self.embedder.encode(tokens)
        self.sow_config.maybe_sow_embeddings(x, self)

        # Create causal mask if not provided
        if attention_mask is None:
            seq_len = tokens.shape[-1]
            attention_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attention_mask = attention_mask[None, :, :]  # Add batch dim

        # Process through layers
        assert attention_mask is not None  # Set above if was None
        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i}"
            layer_cache = cache.get(layer_name) if cache is not None else None
            layer_cache, x = layer(x, positions, layer_cache, attention_mask)
            if new_cache is not None:
                new_cache[layer_name] = layer_cache

        # Final normalization
        x = self.final_norm(x)

        return x, new_cache

    @property
    def embed_dim(self) -> int:
        return self.embedder.embed_dim

    @property
    def num_embed(self) -> int:
        return self.embedder.num_embed

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def init_cache(
        self,
        cache_size: int,
        batch_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> Cache:
        """Initialize KV cache for incremental decoding."""
        return {
            f"layer_{i}": self.layers[i].init_cache(
                cache_size=cache_size,
                batch_size=batch_size,
                dtype=dtype,
            )
            for i in range(self.num_layers)
        }


class Gemma3TTTModel(nnx.Module):
    """Gemma 3 with Test-Time Training (TTT) layer.

    Combines slow weights (frozen Gemma 3 backbone) with fast weights (TTT layer)
    following the PonderTTT architecture:

        output = forward(chunk, theta_slow + theta_fast)

    The TTT layer is applied after the final layer norm as an additive adaptation.
    """

    def __init__(
        self,
        gemma_config: Gemma3Config,
        ttt_config: TTTConfig,
        *,
        rngs: nnx.Rngs,
        tie_word_embeddings: bool = True,
    ):
        """Initialize Gemma 3 TTT model.

        Args:
            gemma_config: Gemma 3 model configuration
            ttt_config: TTT layer configuration
            rngs: Random number generators
            tie_word_embeddings: Share embedding weights with LM head
        """
        self.gemma_config = gemma_config
        self.ttt_config = ttt_config
        self.tie_word_embeddings = tie_word_embeddings

        # Base Gemma 3 model (slow weights - frozen during training)
        self.base_model = Gemma3Model(gemma_config, rngs=rngs)

        # TTT layer pre-norm (following official TTT-LM pattern)
        self.fast_norm = RMSNorm(
            gemma_config.embed_dim,
            rngs=rngs,
            dtype=gemma_config.dtype,
        )

        # Gemma 3 has embed_dim != num_heads * head_dim, so we need projections
        # to match TTT Layer's internal dimension (num_heads * head_dim)
        ttt_internal_dim = ttt_config.num_heads * ttt_config.head_dim
        self._needs_projection = (gemma_config.embed_dim != ttt_internal_dim)

        if self._needs_projection:
            # Project from embed_dim to TTT internal dim
            self.ttt_proj_in = nnx.Linear(
                gemma_config.embed_dim,
                ttt_internal_dim,
                use_bias=False,
                kernel_init=maybe_with_partitioning(
                    nnx.initializers.lecun_normal(),
                    gemma_config.axis_rules,
                    ('embed', 'kv'),
                ),
                rngs=rngs,
            )
            # Project back from TTT internal dim to embed_dim (zero-init for identity at start)
            self.ttt_proj_out = nnx.Linear(
                ttt_internal_dim,
                gemma_config.embed_dim,
                use_bias=False,
                kernel_init=maybe_with_partitioning(
                    nnx.initializers.zeros,
                    gemma_config.axis_rules,
                    ('kv', 'embed'),
                ),
                rngs=rngs,
            )
            # Update TTT config to use internal dimension + axis_rules
            ttt_config = TTTConfig(
                hidden_dim=ttt_internal_dim,
                num_heads=ttt_config.num_heads,
                head_dim=ttt_config.head_dim,
                mini_batch_size=ttt_config.mini_batch_size,
                max_seq_length=ttt_config.max_seq_length,
                dtype=ttt_config.dtype,
                rope_theta=ttt_config.rope_theta,
                axis_rules=gemma_config.axis_rules,
            )
        else:
            # No projection needed, but still pass axis_rules if available
            if gemma_config.axis_rules is not None and ttt_config.axis_rules is None:
                ttt_config = TTTConfig(
                    hidden_dim=ttt_config.hidden_dim,
                    num_heads=ttt_config.num_heads,
                    head_dim=ttt_config.head_dim,
                    mini_batch_size=ttt_config.mini_batch_size,
                    max_seq_length=ttt_config.max_seq_length,
                    dtype=ttt_config.dtype,
                    rope_theta=ttt_config.rope_theta,
                    axis_rules=gemma_config.axis_rules,
                )

        # TTT layer (fast weights - trainable)
        self.fast_layer = TTTLayer(ttt_config, rngs)

        # Optional separate LM head
        if not tie_word_embeddings:
            self.lm_head = nnx.Linear(
                gemma_config.embed_dim,
                gemma_config.num_embed,
                use_bias=False,
                rngs=rngs,
            )

    def __call__(
        self,
        input_ids: Array,
        position_ids: Optional[Array] = None,
        attention_mask: Optional[Array] = None,
        use_ttt: bool = True,
        gating_scale: Optional[Array] = None,
    ) -> dict:
        """Forward pass combining slow and fast weights.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            use_ttt: Whether to apply TTT layer
            gating_scale: Optional scaling for TTT output (for PonderTTT gating)

        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - ttt_stats: TTT layer statistics (if use_ttt=True)
        """
        batch_size, seq_len = input_ids.shape

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(
                batch_size, axis=0
            )

        # Create attention mask
        if attention_mask is None:
            attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attn_mask = attn_mask[None, :, :]
        else:
            # Convert 1D mask to 2D causal mask
            attn_mask = attention_mask[:, None, :] * jnp.tril(
                jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
            )

        # Forward through base model (frozen)
        hidden_states, _ = jax.lax.stop_gradient(
            self.base_model(input_ids, position_ids, None, attn_mask)
        )

        ttt_stats = None

        if use_ttt:
            # Apply TTT layer (following official TTT-LM pattern)
            # 1. Pre-normalization
            hidden_states_normed = self.fast_norm(hidden_states)

            # 2. Apply projection if needed (Gemma 3: embed_dim != num_heads * head_dim)
            if self._needs_projection:
                hidden_states_normed = self.ttt_proj_in(hidden_states_normed)

            # 3. TTT layer
            fast_output, ttt_stats = self.fast_layer(
                hidden_states_normed,
                mask=attention_mask,
                position_ids=position_ids,
                train=True,
                gating_scale=gating_scale,
            )

            # 4. Project back to embed_dim if needed
            if self._needs_projection:
                fast_output = self.ttt_proj_out(fast_output)

            # 5. Residual connection
            hidden_states = hidden_states + fast_output

        # Project to vocabulary
        if self.tie_word_embeddings:
            logits = self.base_model.embedder.decode(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        # Apply final logit softcap if configured
        if self.gemma_config.final_logit_softcap is not None:
            logits = logits / self.gemma_config.final_logit_softcap
            logits = jnp.tanh(logits) * self.gemma_config.final_logit_softcap

        return {
            "logits": logits,
            "ttt_stats": ttt_stats,
        }

    def freeze_base_model(self):
        """Freeze slow weights (pretrained Gemma 3).

        In NNX, freezing is done by filtering parameters during optimizer creation.
        This method is kept for API compatibility.
        """
        pass

    def get_trainable_params(self) -> nnx.State:
        """Get only trainable parameters (TTT layer + fast_norm + projections).

        Returns:
            nnx.State containing only fast-weight layer parameters
        """
        _, fast_state = nnx.split(self.fast_layer)
        _, norm_state = nnx.split(self.fast_norm)
        states = [fast_state, norm_state]

        # Include projection layers if they exist
        if self._needs_projection:
            _, proj_in_state = nnx.split(self.ttt_proj_in)
            _, proj_out_state = nnx.split(self.ttt_proj_out)
            states.extend([proj_in_state, proj_out_state])

        return nnx.State.merge(*states)


def make_causal_attn_mask(input_mask: Array) -> Array:
    """Create causal attention mask from input mask.

    Args:
        input_mask: Input mask [batch, seq_len], True for valid tokens

    Returns:
        Causal attention mask [batch, seq_len, seq_len]
    """
    seq_len = input_mask.shape[-1]
    attn_mask = input_mask[..., None, :]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    attn_mask = attn_mask * causal_mask[None, ...]
    return attn_mask


def build_positions_from_mask(input_mask: Array) -> Array:
    """Compute positions from input mask.

    Args:
        input_mask: Input mask [batch, seq_len], True for valid tokens

    Returns:
        Position IDs [batch, seq_len]
    """
    positions = jnp.cumsum(input_mask, axis=-1)
    return positions - (positions >= 1)
