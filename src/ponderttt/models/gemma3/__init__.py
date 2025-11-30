"""Gemma 3 model implementation for PonderTTT.

This module provides Gemma 3 (4B, 12B) models with Test-Time Training (TTT) support.
Based on the official Flax Gemma implementation with TTT layer integration.
"""

from .layers import Einsum, RMSNorm
from .modules import (
    Attention,
    AttentionType,
    Block,
    Embedder,
    FeedForward,
    DEFAULT_ROPE_BASE_FREQUENCY,
    DEFAULT_ROPE_SCALE_FACTOR,
)
from .positional_embeddings import apply_rope
from .sow_lib import SowConfig
from .config import Gemma3Config
from .model import Gemma3Model, Gemma3TTTModel
from .checkpoint import (
    load_gemma3_from_orbax,
    load_gemma3_from_huggingface,
    save_gemma3_checkpoint,
)
from .sharding import (
    MeshRules,
    ShardingConfig,
    create_device_mesh,
    get_data_sharding,
    setup_sharded_state,
)

__all__ = [
    # Layers
    "Einsum",
    "RMSNorm",
    # Modules
    "Attention",
    "AttentionType",
    "Block",
    "Embedder",
    "FeedForward",
    # Positional
    "apply_rope",
    # Config
    "SowConfig",
    "Gemma3Config",
    "DEFAULT_ROPE_BASE_FREQUENCY",
    "DEFAULT_ROPE_SCALE_FACTOR",
    # Models
    "Gemma3Model",
    "Gemma3TTTModel",
    # Checkpoint
    "load_gemma3_from_orbax",
    "load_gemma3_from_huggingface",
    "save_gemma3_checkpoint",
    # Sharding
    "MeshRules",
    "ShardingConfig",
    "create_device_mesh",
    "get_data_sharding",
    "setup_sharded_state",
]
