"""
Utility functions for PonderTTT.
"""

from .checkpointing import load_checkpoint, save_checkpoint, unwrap_state
from .jax_utils import (
    JaxRNG,
    create_data_sharding,
    create_fsdp_sharding,
    create_mesh,
    create_sharding_constraint,
    cross_entropy_loss,
    per_sample_cross_entropy_loss,
    get_local_batch_size,
    get_metrics,
    init_rng,
    initialize_jax_distributed,
    next_rng,
    print_on_main,
    shard_batch,
)
from .statistics import bootstrap_ci, compute_iqm

__all__ = [
    # Statistics
    "bootstrap_ci",
    "compute_iqm",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "unwrap_state",
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
    "per_sample_cross_entropy_loss",
    "print_on_main",
    "init_rng",
    "next_rng",
]
