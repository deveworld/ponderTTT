# Copyright 2024 PonderTTT Authors.
# Licensed under the Apache License, Version 2.0
#
# Multi-host TPU sharding utilities for Gemma 3.
# Based on official Flax Gemma implementation (MaxText).

"""
Sharding utilities for TPU Pod training.

Supports:
- Single-host multi-device (e.g., 1 host with 8 TPU cores)
- Multi-host multi-device (e.g., TPU v4-64 = 8 hosts × 8 cores)

Usage:
    from ponderttt.models.gemma3.sharding import (
        MeshRules,
        ShardingConfig,
        create_device_mesh,
        setup_sharded_state,
    )

    # Create mesh
    sharding_config = ShardingConfig()
    mesh = create_device_mesh(sharding_config)

    # Initialize with sharding
    state, state_sharding = setup_sharded_state(model, optimizer, mesh)
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

if TYPE_CHECKING:
    from .config import Gemma3Config


@dataclasses.dataclass(frozen=True)
class MeshRules:
    """Sharding rules mapping logical axes to mesh axes.

    These rules define how model parameters are distributed across devices:
    - embed: Embedding dimension sharding (typically FSDP)
    - mlp: MLP layer sharding (typically tensor parallel)
    - kv: Key-value head sharding (typically tensor parallel)
    - vocab: Vocabulary dimension sharding (typically tensor parallel)
    """
    embed: str | None = 'fsdp'
    mlp: str | None = 'tensor'
    kv: str | None = 'tensor'
    vocab: str | None = 'tensor'

    def __call__(self, *logical_axes: str | None) -> P:
        """Convert logical axes to PartitionSpec.

        Args:
            *logical_axes: Sequence of logical axis names (e.g., 'embed', 'mlp')

        Returns:
            PartitionSpec with mesh axis names
        """
        mesh_axes = []
        for axis in logical_axes:
            if axis is None:
                mesh_axes.append(None)
            else:
                mesh_axes.append(getattr(self, axis, None))
        return P(*mesh_axes)


@dataclasses.dataclass
class ShardingConfig:
    """Configuration for multi-host TPU sharding.

    TPU v4-64 example (8 hosts × 8 cores = 64 total):
        - dcn_data_parallelism: 8 (across hosts)
        - ici_fsdp_parallelism: 4 (within host)
        - ici_tensor_parallelism: 2 (within host)

    The product of ICI axes should equal devices per host (typically 8).
    The product of DCN axes should equal number of hosts.
    Use -1 for auto-detection.
    """
    # Mesh axis names
    mesh_axes: tuple[str, ...] = ('data', 'fsdp', 'tensor')

    # Sharding rules
    axis_rules: MeshRules = dataclasses.field(default_factory=MeshRules)

    # Data sharding axes
    data_sharding: tuple[str, ...] = ('data', 'fsdp')

    # DCN (Data Center Network) - Inter-host parallelism
    # -1 means auto-detect based on number of hosts
    dcn_data_parallelism: int = -1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1

    # ICI (Inter-Chip Interconnection) - Intra-host parallelism
    # -1 means auto-detect based on devices per host
    ici_data_parallelism: int = 1
    ici_fsdp_parallelism: int = -1
    ici_tensor_parallelism: int = 1


def _fill_unspecified_mesh_axes(
    parallelism_vals: list[int],
    target_product: int,
    parallelism_type: str,
) -> list[int]:
    """Fill in unspecified (-1) parallelism values.

    Args:
        parallelism_vals: List of parallelism values, -1 for unspecified
        target_product: Target product (num_slices for DCN, devices_per_slice for ICI)
        parallelism_type: 'DCN' or 'ICI' for error messages

    Returns:
        List with -1 replaced by computed value
    """
    if -1 in parallelism_vals:
        if parallelism_vals.count(-1) != 1:
            raise ValueError(
                f"Found unspecified values (-1) for more than one {parallelism_type} "
                "parallelism axis. At most one axis can be unspecified."
            )

        # Compute the unspecified value
        specified_product = -np.prod(parallelism_vals)  # Negate because one is -1
        determined_val = target_product / specified_product

        if determined_val < 1 or not float(determined_val).is_integer():
            raise ValueError(
                f"Unable to determine unspecified {parallelism_type} parallelism value. "
                f"Target: {target_product}, specified product: {specified_product}"
            )

        parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

    actual_product = np.prod(parallelism_vals)
    target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'

    if actual_product != target_product:
        raise ValueError(
            f"Number of {target_type} ({target_product}) does not match "
            f"the product of {parallelism_type} parallelism ({actual_product})"
        )

    return parallelism_vals


def create_device_mesh(config: ShardingConfig) -> Mesh:
    """Create device mesh for multi-host TPU training.

    Automatically detects:
    - Number of hosts (slices)
    - Devices per host
    - Whether running in multi-slice environment

    Args:
        config: Sharding configuration

    Returns:
        JAX Mesh object for distributed training
    """
    devices = jax.devices()
    num_devices = len(devices)

    # Detect number of slices (hosts)
    try:
        num_slices = 1 + max(d.slice_index for d in devices)
    except AttributeError:
        num_slices = 1

    num_devices_per_slice = num_devices // num_slices

    logging.info(f"Devices: {num_devices} total, {num_slices} slices, "
                 f"{num_devices_per_slice} per slice")

    # Check for multi-slice environment
    multi_slice_env = hasattr(devices[0], 'slice_index')

    # Build parallelism configurations
    dcn_parallelism = [
        config.dcn_data_parallelism,
        config.dcn_fsdp_parallelism,
        config.dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_tensor_parallelism,
    ]

    # Fill unspecified values
    dcn_parallelism = _fill_unspecified_mesh_axes(
        dcn_parallelism, num_slices, 'DCN'
    )
    ici_parallelism = _fill_unspecified_mesh_axes(
        ici_parallelism, num_devices_per_slice, 'ICI'
    )

    # Create mesh
    if multi_slice_env:
        devices_array = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism, dcn_parallelism
        )
    else:
        devices_array = mesh_utils.create_device_mesh(ici_parallelism)

    mesh = Mesh(devices_array, config.mesh_axes)

    logging.info(f"Created mesh: {mesh}")
    logging.info(f"Mesh shape: {mesh.shape}")

    return mesh


def get_data_sharding(mesh: Mesh, config: ShardingConfig) -> NamedSharding:
    """Get sharding specification for input data.

    Args:
        mesh: Device mesh
        config: Sharding configuration

    Returns:
        NamedSharding for input batches
    """
    return NamedSharding(mesh, P(*config.data_sharding))


def setup_sharded_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    mesh: Mesh,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize model and optimizer state with sharding.

    Args:
        model: NNX model (already initialized)
        optimizer: NNX optimizer
        mesh: Device mesh

    Returns:
        Tuple of (state_dict, state_sharding)
    """
    def _to_array(x):
        if not isinstance(x, jax.Array):
            x = jnp.asarray(x)
        return x

    @jax.jit
    def shard_state(model_state, opt_state):
        # Convert to arrays
        model_state = jax.tree.map(_to_array, model_state)
        opt_state = jax.tree.map(_to_array, opt_state)

        # Get partition specs from NNX annotations
        model_spec = nnx.get_partition_spec(model_state)
        opt_spec = nnx.get_partition_spec(opt_state)

        # Apply sharding constraints
        model_state = jax.lax.with_sharding_constraint(model_state, model_spec)
        opt_state = jax.lax.with_sharding_constraint(opt_state, opt_spec)

        return model_state, opt_state

    # Get current state
    model_state = nnx.state(model)
    opt_state = nnx.state(optimizer)

    # Shard within mesh context
    with jax.set_mesh(mesh):
        model_state, opt_state = shard_state(model_state, opt_state)

    # Get named shardings for jit
    model_sharding = nnx.get_named_sharding(model_state, mesh)
    opt_sharding = nnx.get_named_sharding(opt_state, mesh)

    state = {"model": model_state, "optimizer": opt_state}
    state_sharding = {"model": model_sharding, "optimizer": opt_sharding}

    return state, state_sharding


def shard_params_for_ttt(
    params: nnx.State,
    mesh: Mesh,
    rules: MeshRules,
) -> nnx.State:
    """Apply sharding to TTT layer parameters.

    TTT layer parameters need special handling because they're not
    part of the base model's sharding annotations.

    Args:
        params: TTT layer parameters
        mesh: Device mesh
        rules: Sharding rules

    Returns:
        Sharded parameters
    """
    def _to_array(x):
        if not isinstance(x, jax.Array):
            x = jnp.asarray(x)
        return x

    @jax.jit
    def shard_fn(params):
        params = jax.tree.map(_to_array, params)
        # TTT params: replicate across all devices (no sharding)
        # This is safe because TTT layer is small relative to base model
        return params

    with jax.set_mesh(mesh):
        params = shard_fn(params)

    return params
