"""
Test distributed JAX setup for TPU.

This script tests:
1. JAX distributed initialization
2. Mesh creation
3. Data sharding
4. Multi-host coordination

Usage:
    # Single host test
    python scripts/test_distributed.py

    # Multi-host test (run on all hosts)
    python scripts/test_distributed.py --multi_host
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import argparse
from ponderttt.utils import (
    initialize_jax_distributed,
    create_mesh,
    shard_batch,
    get_local_batch_size,
    print_on_main,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_host", action="store_true")
    return parser.parse_args()


def test_basic_jax():
    """Test basic JAX functionality."""
    print("\n[Test 1/5] Basic JAX")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Device count: {jax.device_count()}")
    print(f"  Local device count: {jax.local_device_count()}")

    try:
        print(f"  Process index: {jax.process_index()}")
        print(f"  Process count: {jax.process_count()}")
    except:
        print(f"  ⚠️ Multi-host not initialized")

    print("  ✓ Basic JAX works")


def test_mesh_creation():
    """Test mesh creation."""
    print("\n[Test 2/5] Mesh Creation")

    # Try different mesh shapes
    device_count = jax.device_count()
    print(f"  Total devices: {device_count}")

    if device_count == 8:
        # Single TPU host
        mesh = create_mesh((8, 1), ('batch', 'model'))
    elif device_count == 64:
        # TPU v4-64
        mesh = create_mesh((64, 1), ('batch', 'model'))
    else:
        # Generic
        mesh = create_mesh((device_count, 1), ('batch', 'model'))

    print(f"  Mesh shape: {mesh.devices.shape}")
    print(f"  Mesh axes: {mesh.axis_names}")
    print("  ✓ Mesh creation works")

    return mesh


def test_data_sharding(mesh):
    """Test data sharding."""
    print("\n[Test 3/5] Data Sharding")

    # Create dummy batch
    global_batch_size = 64
    seq_len = 128
    batch = {
        'input_ids': jnp.ones((global_batch_size, seq_len), dtype=jnp.int32),
        'attention_mask': jnp.ones((global_batch_size, seq_len), dtype=jnp.bool_),
    }

    print(f"  Original batch shape: {batch['input_ids'].shape}")

    # Shard batch
    sharded_batch = shard_batch(batch, mesh, batch_axis='batch')

    print(f"  Sharded batch shape: {sharded_batch['input_ids'].shape}")
    print(f"  Sharding: {sharded_batch['input_ids'].sharding}")
    print("  ✓ Data sharding works")


def test_collective_ops(mesh):
    """Test collective operations."""
    print("\n[Test 4/5] Collective Operations")

    # Create local data
    local_data = jnp.ones(jax.local_device_count()) * jax.process_index()

    # pmap all-reduce
    @jax.pmap
    def all_reduce_sum(x):
        return jax.lax.psum(x, 'i')

    result = all_reduce_sum(local_data)
    print(f"  Local data: {local_data}")
    print(f"  All-reduce result: {result}")
    print("  ✓ Collective operations work")


def test_batch_size_calculation():
    """Test batch size calculations."""
    print("\n[Test 5/5] Batch Size Calculation")

    global_batch_sizes = [256, 512, 1024]

    for gbs in global_batch_sizes:
        local_bs = get_local_batch_size(gbs)
        print(f"  Global: {gbs:4d} -> Local: {local_bs:4d}")

    print("  ✓ Batch size calculation works")


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Distributed JAX Test")
    print("=" * 60)

    # Initialize if multi-host
    if args.multi_host:
        print("\nInitializing multi-host JAX...")
        initialize_jax_distributed()
    else:
        print("\nSingle-host mode (no initialization needed)")

    # Run tests
    test_basic_jax()
    mesh = test_mesh_creation()
    test_data_sharding(mesh)
    test_collective_ops(mesh)
    test_batch_size_calculation()

    print("\n" + "=" * 60)
    print_on_main("✓ All distributed tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
