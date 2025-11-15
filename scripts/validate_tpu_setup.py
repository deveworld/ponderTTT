"""
Validation script for TPU multi-host setup.

This script performs comprehensive checks to ensure your TPU Pod
configuration is correct before running expensive training jobs.

Usage:
    # Single host (8 TPU chips)
    python scripts/validate_tpu_setup.py

    # Multi-host (e.g., TPU v4-64: 8 hosts × 8 chips)
    # Run on ALL hosts simultaneously
    gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
        --zone=us-central2-b \
        --worker=all \
        --command="cd ~/ponderttt && python scripts/validate_tpu_setup.py --multi_host"

What this script checks:
1. JAX distributed initialization
2. Device mesh creation
3. Data sharding across hosts
4. Parameter sharding
5. Gradient computation and aggregation
6. Cross-host collective operations
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import argparse
from functools import partial

from ponderttt.utils import (
    initialize_jax_distributed,
    create_mesh,
    shard_batch,
    create_sharding_constraint,
    print_on_main,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TPU setup")
    parser.add_argument(
        "--multi_host",
        action="store_true",
        help="Enable multi-host validation"
    )
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="8,1",
        help="Mesh shape as 'batch,model'"
    )
    return parser.parse_args()


def check_jax_distributed(multi_host: bool):
    """Check JAX distributed initialization."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 1: JAX Distributed Initialization")
    print_on_main("=" * 80)

    if multi_host:
        try:
            initialize_jax_distributed()
            print_on_main("✅ PASS: Multi-host JAX initialized")
        except Exception as e:
            print_on_main(f"❌ FAIL: {e}")
            return False
    else:
        print_on_main("✅ PASS: Single-host mode")

    # Print device information
    print_on_main(f"  Process index: {jax.process_index()}")
    print_on_main(f"  Process count: {jax.process_count()}")
    print_on_main(f"  Local devices: {jax.local_device_count()}")
    print_on_main(f"  Global devices: {jax.device_count()}")

    # Verify device count
    expected_local = 8  # TPU v4 has 8 chips per host
    if jax.local_device_count() != expected_local:
        print_on_main(f"⚠️  WARNING: Expected {expected_local} local devices, got {jax.local_device_count()}")

    return True


def check_mesh_creation(mesh_shape_str: str):
    """Check device mesh creation."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 2: Device Mesh Creation")
    print_on_main("=" * 80)

    try:
        mesh_shape = tuple(map(int, mesh_shape_str.split(",")))
        mesh = create_mesh(mesh_shape, ('batch', 'model'))
        print_on_main(f"✅ PASS: Mesh created with shape {mesh_shape}")
        print_on_main(f"  Mesh axes: {mesh.axis_names}")
        print_on_main(f"  Total devices in mesh: {mesh.devices.size}")

        # Verify mesh size matches device count
        if mesh.devices.size != jax.device_count():
            print_on_main(f"❌ FAIL: Mesh size {mesh.devices.size} != device count {jax.device_count()}")
            return None

        return mesh
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        return None


def check_data_sharding(mesh):
    """Check data batch sharding."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 3: Data Batch Sharding")
    print_on_main("=" * 80)

    try:
        # Create dummy batch
        batch_size = 64
        seq_length = 128
        batch = {
            'input_ids': jnp.ones((batch_size, seq_length), dtype=jnp.int32),
            'attention_mask': jnp.ones((batch_size, seq_length), dtype=jnp.int32),
        }

        # Shard batch
        sharded_batch = shard_batch(batch, mesh, batch_axis='batch')

        print_on_main(f"✅ PASS: Batch sharded successfully")
        print_on_main(f"  Original shape: {batch['input_ids'].shape}")
        print_on_main(f"  Sharded shape: {sharded_batch['input_ids'].shape}")

        # Check sharding
        if hasattr(sharded_batch['input_ids'], 'sharding'):
            print_on_main(f"  Sharding spec: {sharded_batch['input_ids'].sharding.spec}")

        return sharded_batch
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_parameter_sharding(mesh):
    """Check parameter sharding."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 4: Parameter Sharding")
    print_on_main("=" * 80)

    try:
        from jax.sharding import NamedSharding, PartitionSpec as P

        # Create dummy parameters
        params = {
            'embedding': jnp.ones((50000, 768)),  # Large embedding
            'layer_0': {
                'kernel': jnp.ones((768, 3072)),  # Large weight matrix
                'bias': jnp.ones((3072,)),         # Small bias
            }
        }

        # Apply sharding
        from ponderttt.models import apply_sharding_to_params
        sharded_params = apply_sharding_to_params(params, mesh)

        print_on_main(f"✅ PASS: Parameters sharded successfully")

        # Inspect sharding
        from ponderttt.models import inspect_sharding
        inspect_sharding(sharded_params, max_params=10)

        return sharded_params
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_gradient_computation(mesh, sharded_params):
    """Check gradient computation and aggregation."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 5: Gradient Computation and Aggregation")
    print_on_main("=" * 80)

    try:
        from jax.lax import with_sharding_constraint

        # Simple loss function
        def loss_fn(params):
            # Compute simple loss from embedding
            embedding = params['embedding']
            return jnp.mean(embedding ** 2)

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(sharded_params)

        # Apply sharding constraint to gradients
        param_sharding = create_sharding_constraint(mesh, 'batch')
        grads = jax.tree_map(
            lambda g: with_sharding_constraint(g, param_sharding) if g is not None else None,
            grads
        )

        print_on_main(f"✅ PASS: Gradients computed successfully")
        print_on_main(f"  Loss: {float(loss):.6f}")
        print_on_main(f"  Gradient shape (embedding): {grads['embedding'].shape}")

        return True
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_collective_ops(mesh):
    """Check cross-host collective operations."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 6: Collective Operations (AllReduce)")
    print_on_main("=" * 80)

    try:
        # Create a value unique to each process
        local_value = jnp.array([float(jax.process_index())])

        # AllReduce (sum across all processes)
        @jax.jit
        def allreduce_sum(x):
            return jax.lax.psum(x, axis_name='batch')

        with mesh:
            # Shard the value
            from jax.sharding import NamedSharding, PartitionSpec as P
            sharding = NamedSharding(mesh, P('batch'))
            sharded_value = jax.device_put(local_value, sharding)

            # Perform allreduce
            result = allreduce_sum(sharded_value)

        expected = sum(range(jax.process_count()))
        actual = float(result[0])

        if abs(actual - expected) < 1e-6:
            print_on_main(f"✅ PASS: AllReduce works correctly")
            print_on_main(f"  Expected sum: {expected}")
            print_on_main(f"  Actual sum: {actual}")
        else:
            print_on_main(f"❌ FAIL: AllReduce incorrect")
            print_on_main(f"  Expected: {expected}, Got: {actual}")
            return False

        return True
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_jit_compilation(mesh):
    """Check JIT compilation with sharding."""
    print_on_main("\n" + "=" * 80)
    print_on_main("TEST 7: JIT Compilation with Sharding")
    print_on_main("=" * 80)

    try:
        # Define a simple jitted function
        @partial(jax.jit)
        def simple_fn(x):
            return x * 2 + 1

        # Create sharded input
        x = jnp.ones((64, 128))
        sharded_x = shard_batch({'x': x}, mesh, batch_axis='batch')['x']

        # Run jitted function
        result = simple_fn(sharded_x)

        print_on_main(f"✅ PASS: JIT compilation works with sharded data")
        print_on_main(f"  Input shape: {sharded_x.shape}")
        print_on_main(f"  Output shape: {result.shape}")

        return True
    except Exception as e:
        print_on_main(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    print_on_main("=" * 80)
    print_on_main("PonderTTT TPU Setup Validation")
    print_on_main("=" * 80)
    print_on_main(f"Multi-host mode: {args.multi_host}")
    print_on_main(f"Mesh shape: {args.mesh_shape}")

    results = []

    # Test 1: JAX distributed
    results.append(("JAX Distributed", check_jax_distributed(args.multi_host)))

    # Test 2: Mesh creation
    mesh = check_mesh_creation(args.mesh_shape)
    results.append(("Mesh Creation", mesh is not None))

    if mesh is None:
        print_on_main("\n❌ Cannot continue: Mesh creation failed")
        sys.exit(1)

    # Test 3: Data sharding
    sharded_batch = check_data_sharding(mesh)
    results.append(("Data Sharding", sharded_batch is not None))

    # Test 4: Parameter sharding
    sharded_params = check_parameter_sharding(mesh)
    results.append(("Parameter Sharding", sharded_params is not None))

    # Test 5: Gradient computation
    if sharded_params is not None:
        grad_ok = check_gradient_computation(mesh, sharded_params)
        results.append(("Gradient Computation", grad_ok))

    # Test 6: Collective operations
    collective_ok = check_collective_ops(mesh)
    results.append(("Collective Ops", collective_ok))

    # Test 7: JIT compilation
    jit_ok = check_jit_compilation(mesh)
    results.append(("JIT Compilation", jit_ok))

    # Summary
    print_on_main("\n" + "=" * 80)
    print_on_main("VALIDATION SUMMARY")
    print_on_main("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print_on_main(f"  {test_name:30s} {status}")
        if not passed:
            all_passed = False

    print_on_main("=" * 80)

    if all_passed:
        print_on_main("🎉 All tests passed! Your TPU setup is ready for training.")
        print_on_main("\nNext steps:")
        print_on_main("  python scripts/train_tpu.py --multi_host --mesh_shape='64,1'")
    else:
        print_on_main("⚠️  Some tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
