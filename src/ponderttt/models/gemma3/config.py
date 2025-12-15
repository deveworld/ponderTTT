"""Gemma 3 configuration for PonderTTT.

Based on the official Flax Gemma TransformerConfig with additions for TTT integration.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import jax.numpy as jnp

from .modules import AttentionType, DEFAULT_ROPE_BASE_FREQUENCY, DEFAULT_ROPE_SCALE_FACTOR


class QueryPreAttentionNormalisation(enum.Enum):
    """Query pre-attention normalization strategy."""

    BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()
    BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()
    BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


# Gemma 3 attention pattern: 5 local + 1 global
GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def make_attention_layers_types(
    pattern: tuple[AttentionType, ...],
    num_layers: int,
) -> tuple[AttentionType, ...]:
    """Returns the list of attention types for every layer."""
    pattern_size = len(pattern)
    out = pattern * (num_layers // pattern_size)
    if num_layers % pattern_size != 0:
        out += pattern[: num_layers % pattern_size]
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class Gemma3Config:
    """Configuration for Gemma 3 model.

    Based on official Flax Gemma TransformerConfig with PonderTTT extensions.
    """

    # Model architecture
    num_layers: int
    num_embed: int  # Vocabulary size
    embed_dim: int  # Hidden dimension
    hidden_dim: int  # MLP intermediate dimension
    num_heads: int  # Number of attention heads
    head_dim: int  # Dimension per head
    num_kv_heads: int  # Number of key-value heads (for GQA)

    # Attention configuration
    attention_types: tuple[AttentionType, ...]
    sliding_window_size: int = 1024
    use_qk_norm: bool = True
    query_pre_attn_norm: QueryPreAttentionNormalisation = (
        QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
    )
    attn_logits_soft_cap: float | None = None
    final_logit_softcap: float | None = None

    # Normalization
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True

    # RoPE configuration
    local_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
    global_base_frequency: int = 1_000_000
    local_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
    global_scale_factor: float = 8.0

    # Other
    transpose_gating_einsum: bool = True
    dtype: Any = jnp.bfloat16
    axis_rules: Any | None = None

    def query_pre_attn_scalar(self) -> float:
        """Returns the scalar to multiply the query by before attention."""
        match self.query_pre_attn_norm:
            case QueryPreAttentionNormalisation.BY_EMBED_DIM_DIV_NUM_HEADS:
                return self.embed_dim // self.num_heads
            case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:
                return (self.embed_dim // self.num_heads) ** -0.5
            case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM | _:
                return self.head_dim ** -0.5

    @classmethod
    def gemma3_4b(cls, dtype: Any = jnp.bfloat16, **override) -> Gemma3Config:
        """Gemma 3 4B configuration.

        Model specs:
        - Parameters: ~4B
        - Layers: 34
        - Hidden: 2560
        - Heads: 8 (4 KV heads for GQA)
        """
        num_layers = 34
        config = {
            "num_layers": num_layers,
            "num_embed": 262_144,
            "embed_dim": 2560,
            "hidden_dim": 2560 * 8 // 2,  # 10240
            "num_heads": 8,
            "head_dim": 256,
            "num_kv_heads": 4,
            "attention_types": make_attention_layers_types(
                GEMMA3_ATTENTION_PATTERN, num_layers
            ),
            "sliding_window_size": 1024,
            "use_qk_norm": True,
            "use_post_attn_norm": True,
            "use_post_ffw_norm": True,
            "query_pre_attn_norm": QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
            "attn_logits_soft_cap": None,
            "final_logit_softcap": None,
            "local_base_frequency": 10_000,
            "global_base_frequency": 1_000_000,
            "global_scale_factor": 8.0,
            "transpose_gating_einsum": True,
            "dtype": dtype,
        }
        config.update(override)
        return cls(**config)  # type: ignore[arg-type]

    @classmethod
    def gemma3_12b(cls, dtype: Any = jnp.bfloat16, **override) -> Gemma3Config:
        """Gemma 3 12B configuration.

        Model specs:
        - Parameters: ~12B
        - Layers: 48
        - Hidden: 3840
        - Heads: 16 (8 KV heads for GQA)
        """
        num_layers = 48
        config = {
            "num_layers": num_layers,
            "num_embed": 262_144,
            "embed_dim": 30 * 128,  # 3840
            "hidden_dim": 8 * 30 * 128 // 2,  # 15360
            "num_heads": 16,
            "head_dim": 256,
            "num_kv_heads": 8,
            "attention_types": make_attention_layers_types(
                GEMMA3_ATTENTION_PATTERN, num_layers
            ),
            "sliding_window_size": 1024,
            "use_qk_norm": True,
            "use_post_attn_norm": True,
            "use_post_ffw_norm": True,
            "query_pre_attn_norm": QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
            "attn_logits_soft_cap": None,
            "final_logit_softcap": None,
            "local_base_frequency": 10_000,
            "global_base_frequency": 1_000_000,
            "global_scale_factor": 8.0,
            "transpose_gating_einsum": True,
            "dtype": dtype,
        }
        config.update(override)
        return cls(**config)  # type: ignore[arg-type]

    @classmethod
    def gemma3_1b(cls, dtype: Any = jnp.bfloat16, **override) -> Gemma3Config:
        """Gemma 3 1B configuration (for testing)."""
        num_layers = 26
        config = {
            "num_layers": num_layers,
            "num_embed": 262_144,
            "embed_dim": 1152,
            "hidden_dim": 6 * 1152,  # 6912
            "num_heads": 4,
            "head_dim": 256,
            "num_kv_heads": 1,
            "attention_types": make_attention_layers_types(
                GEMMA3_ATTENTION_PATTERN, num_layers
            ),
            "sliding_window_size": 512,
            "use_qk_norm": True,
            "use_post_attn_norm": True,
            "use_post_ffw_norm": True,
            "query_pre_attn_norm": QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
            "attn_logits_soft_cap": None,
            "final_logit_softcap": None,
            "local_base_frequency": 10_000,
            "global_base_frequency": 1_000_000,
            "global_scale_factor": 8.0,
            "transpose_gating_einsum": True,
            "dtype": dtype,
        }
        config.update(override)
        return cls(**config)  # type: ignore[arg-type]

    def __post_init__(self):
        if self.num_heads != self.num_kv_heads:
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"Number of query heads ({self.num_heads}) must be divisible by "
                    f"number of key/value heads ({self.num_kv_heads})."
                )
