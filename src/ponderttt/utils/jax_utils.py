"""
JAX distributed training utilities for TPU multi-host setup.

Based on:
- Google Cloud TPU Pods documentation
- TTT-LM-JAX implementation
- Modern JAX patterns (jax.make_mesh, NamedSharding)
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as PS


class JaxRNG:
    """
    Convenient stateful JAX RNG wrapper.
    Can be used to wrap RNG inside pure functions.
    """

    @classmethod
    def from_seed(cls, seed: int):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return dict(zip(keys, split_rngs[1:]))


def initialize_jax_distributed(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
) -> None:
    """
    Initialize JAX for multi-host distributed training.

    For TPU Pods, this should be called at the start of your program
    on each host before any JAX operations.

    Args:
        coordinator_address: Address of coordinator (e.g., "10.0.0.1:1234")
                           If None, uses environment variables
        num_processes: Total number of hosts in the TPU Pod
                      If None, uses environment variables
        process_id: ID of this process (0 to num_processes-1)
                   If None, uses environment variables

    Note:
        On Google Cloud TPU Pods, you can often just call:
        jax.distributed.initialize()
        and JAX will automatically detect the configuration.
    """
    if coordinator_address is None and num_processes is None:
        # Simple initialization - JAX auto-detects TPU Pod config
        try:
            jax.distributed.initialize()
            print(" JAX distributed initialized automatically")
        except RuntimeError:
            # JAX distributed not available (single host or already initialized)
            print(" JAX distributed initialization skipped (single host?)")
            return
    else:
        # Explicit initialization
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
        print(" JAX distributed initialized explicitly")

    # Print device info (but only from process 0 to avoid spam)
    if jax.process_index() == 0:
        print(f"Total devices: {jax.device_count()}")
        print(f"Local devices per host: {jax.local_device_count()}")
        print(f"Number of hosts: {jax.process_count()}")


def create_mesh(
    mesh_shape: tuple[int, ...],
    axis_names: tuple[str, ...],
) -> Mesh:
    """
    Create a device mesh for distributed training.

    Args:
        mesh_shape: Shape of the mesh, e.g., (8, 1) for 8-way data parallel
        axis_names: Names for mesh axes, e.g., ('batch', 'model')

    Returns:
        JAX Mesh object

    Example:
        # For TPU v4-64 with 8 hosts × 8 chips = 64 devices
        # Data parallel only:
        mesh = create_mesh((64, 1), ('batch', 'model'))

        # 8-way data parallel, 8-way FSDP:
        mesh = create_mesh((8, 8), ('dp', 'fsdp'))
    """
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices, axis_names=axis_names)

    if jax.process_index() == 0:
        print(f" Created mesh with shape {mesh_shape} and axes {axis_names}")
        print(f"  Total devices in mesh: {mesh.devices.size}")

    return mesh


def create_data_sharding(
    mesh: Mesh,
    batch_axis: str = 'batch',
) -> NamedSharding:
    """
    Create sharding specification for data batches.

    Args:
        mesh: JAX Mesh
        batch_axis: Name of batch axis in mesh

    Returns:
        NamedSharding for data batches
    """
    return NamedSharding(mesh, PS(batch_axis, None))


def create_fsdp_sharding(
    mesh: Mesh,
    fsdp_axis: str = 'fsdp',
) -> NamedSharding:
    """
    Create FSDP sharding specification for model parameters.

    Args:
        mesh: JAX Mesh
        fsdp_axis: Name of FSDP axis in mesh

    Returns:
        NamedSharding for FSDP parameters
    """
    return NamedSharding(mesh, PS(fsdp_axis, None))


def create_sharding_constraint(
    mesh: Mesh,
    axis: str | None = None,
    replicated: bool = False,
) -> NamedSharding:
    """
    Create sharding constraint for parameters and gradients.

    Used with jax.lax.with_sharding_constraint to control
    how tensors are sharded across devices in multi-host setups.

    Args:
        mesh: JAX Mesh
        axis: Mesh axis to shard along (e.g., 'batch', 'model', 'fsdp')
              If None and replicated=False, shards along first axis
        replicated: If True, replicate across all devices (no sharding)

    Returns:
        NamedSharding for use with with_sharding_constraint

    Examples:
        # Replicate parameters
        sharding = create_sharding_constraint(mesh, replicated=True)
        params = with_sharding_constraint(params, sharding)

        # Shard along batch axis
        sharding = create_sharding_constraint(mesh, 'batch')
        grads = with_sharding_constraint(grads, sharding)

        # Shard along model axis
        sharding = create_sharding_constraint(mesh, 'model')
        params = with_sharding_constraint(params, sharding)
    """
    if replicated:
        # Replicate across all devices
        return NamedSharding(mesh, PS())
    elif axis is not None:
        # Shard along specified axis
        return NamedSharding(mesh, PS(axis))
    else:
        # Default: shard along first mesh axis
        first_axis = mesh.axis_names[0]
        return NamedSharding(mesh, PS(first_axis))


def shard_batch(
    batch: Any,
    mesh: Mesh,
    batch_axis: str = 'batch',
) -> Any:
    """
    Shard a batch across devices.

    Args:
        batch: Batch to shard (can be dict, tuple, or array)
        mesh: JAX Mesh
        batch_axis: Name of batch axis in mesh

    Returns:
        Sharded batch
    """
    sharding = create_data_sharding(mesh, batch_axis)

    def shard_array(x):
        if isinstance(x, jnp.ndarray):
            return jax.device_put(x, sharding)
        return x

    return jax.tree_map(shard_array, batch)


def get_local_batch_size(
    global_batch_size: int,
) -> int:
    """
    Calculate local batch size for this host.

    Args:
        global_batch_size: Total batch size across all devices

    Returns:
        Batch size for this host

    Example:
        # For global_batch_size=512 on 8 hosts with 8 chips each (64 devices):
        # per_device_batch = 512 / 64 = 8
        # local_batch = 8 × 8 = 64 per host
    """
    num_devices = jax.device_count()
    per_device_batch = global_batch_size // num_devices

    local_devices = jax.local_device_count()
    local_batch = per_device_batch * local_devices

    if jax.process_index() == 0:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-device batch size: {per_device_batch}")
        print(f"Local batch size per host: {local_batch}")

    return local_batch


def get_metrics(metrics, unreplicate: bool = False):
    """
    Get metrics from devices.

    Args:
        metrics: Metrics dict
        unreplicate: Whether to unreplicate from pmap

    Returns:
        Metrics as Python dict
    """
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)

    metrics = jax.device_get(metrics)
    return {key: float(val) for key, val in metrics.items()}


def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Compute cross-entropy loss.

    Args:
        logits: Shape (batch, seq_len, vocab_size)
        labels: Shape (batch, seq_len)
        mask: Optional mask, shape (batch, seq_len)

    Returns:
        Scalar loss
    """
    if mask is None:
        mask = jnp.ones(labels.shape[:2])

    # Ensure mask is float
    mask = mask.astype(jnp.float32)

    # Valid sequence length
    valid_length = jnp.maximum(jnp.sum(mask, axis=-1), 1e-10)

    # Log probabilities
    logits = logits.astype(jnp.float32)  # for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Gather log probs for true labels
    token_log_probs = jnp.take_along_axis(
        log_probs,
        jnp.expand_dims(labels, -1),
        axis=-1
    ).squeeze(-1)

    # Mask and average
    token_log_probs = jnp.where(mask > 0, token_log_probs, 0.0)
    loss = -jnp.mean(jnp.sum(token_log_probs, axis=-1) / valid_length)

    return loss


def print_on_main(message: str) -> None:
    """
    Print message only from main process to avoid spam.

    Args:
        message: Message to print
    """
    if jax.process_index() == 0:
        print(message)


# Global RNG for convenience
_global_rng = None


def init_rng(seed: int) -> None:
    """Initialize global RNG."""
    global _global_rng
    _global_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    """Get next RNG split."""
    global _global_rng
    if _global_rng is None:
        init_rng(42)
    return _global_rng(*args, **kwargs)
