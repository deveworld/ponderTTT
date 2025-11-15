"""
Utility functions for PonderTTT.
"""

from .features import extract_features, FeatureExtractor
from .statistics import bootstrap_ci, compute_iqm
from .checkpointing import save_checkpoint, load_checkpoint
from .jax_utils import (
    JaxRNG,
    initialize_jax_distributed,
    create_mesh,
    create_data_sharding,
    create_fsdp_sharding,
    create_sharding_constraint,
    shard_batch,
    get_local_batch_size,
    get_metrics,
    cross_entropy_loss,
    print_on_main,
    init_rng,
    next_rng,
)

__all__ = [
    # Features
    "extract_features",
    "FeatureExtractor",
    # Statistics
    "bootstrap_ci",
    "compute_iqm",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    # JAX distributed utilities
    "JaxRNG",
    "initialize_jax_distributed",
    "create_mesh",
    "create_data_sharding",
    "create_fsdp_sharding",
    "create_sharding_constraint",
    "shard_batch",
    "get_local_batch_size",
    "get_metrics",
    "cross_entropy_loss",
    "print_on_main",
    "init_rng",
    "next_rng",
]
