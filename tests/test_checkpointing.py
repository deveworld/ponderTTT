"""
Test script to verify checkpointing functionality.
"""

import tempfile
from pathlib import Path
import jax.numpy as jnp
import optax

from ponderttt.utils.checkpointing import (
    get_latest_checkpoint_step,
    load_checkpoint,
    save_checkpoint,
    wait_for_checkpoints,
)

def test_basic_save_load():
    """Test basic save and load functionality."""
    print("=" * 60)
    print("Test 1: Basic Save/Load")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Create some dummy state
        params = {
            "layer1": {"weights": jnp.ones((3, 3)), "bias": jnp.zeros(3)},
            "layer2": {"weights": jnp.ones((3, 2))},
        }

        opt = optax.adam(1e-3)
        opt_state = opt.init(params)

        metadata = {
            "chunk_count": 42,
            "total_cost": 123.45,
            "total_loss": 67.89,
            "results": {"chunks": [{"id": 1, "loss": 0.5}]},
        }

        # Save checkpoint
        print("\n1. Saving checkpoint...")
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=42,
            state={"params": params, "opt_state": opt_state},
            metadata=metadata,
        )
        wait_for_checkpoints()  # Wait for async save to complete
        print("   OK Checkpoint saved")

        # Check latest step
        print("\n2. Getting latest checkpoint step...")
        latest = get_latest_checkpoint_step(checkpoint_dir)
        print(f"   OK Latest step: {latest}")
        assert latest == 42, f"Expected 42, got {latest}"

        # Load checkpoint
        print("\n3. Loading checkpoint...")
        loaded = load_checkpoint(checkpoint_dir, step=42)
        print("   OK Checkpoint loaded")

        # Verify contents
        print("\n4. Verifying loaded data...")
        assert loaded["step"] == 42
        assert loaded["metadata"]["chunk_count"] == 42
        assert loaded["metadata"]["total_cost"] == 123.45
        assert loaded["metadata"]["total_loss"] == 67.89
        assert len(loaded["metadata"]["results"]["chunks"]) == 1

        # Verify params shape
        assert loaded["state"]["params"]["layer1"]["weights"].shape == (3, 3)
        assert loaded["state"]["params"]["layer1"]["bias"].shape == (3,)
        print("   OK All data verified")

    print("\n" + "=" * 60)
    print("Test 1 PASSED")
    print("=" * 60)


def test_multiple_checkpoints():
    """Test saving multiple checkpoints."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple Checkpoints")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        params = {"weights": jnp.ones((2, 2))}
        opt = optax.adam(1e-3)
        opt_state = opt.init(params)

        # Save multiple checkpoints
        print("\n1. Saving checkpoints at steps 10, 20, 30...")
        for step in [10, 20, 30]:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                state={"params": params, "opt_state": opt_state},
                metadata={"step_num": step, "value": step * 10},
            )
        wait_for_checkpoints()  # Wait for async saves to complete
        print("   OK All checkpoints saved")

        # Get latest
        print("\n2. Getting latest checkpoint...")
        latest = get_latest_checkpoint_step(checkpoint_dir)
        print(f"   OK Latest step: {latest}")
        assert latest == 30, f"Expected 30, got {latest}"

        # Load specific checkpoint
        print("\n3. Loading checkpoint at step 20...")
        loaded = load_checkpoint(checkpoint_dir, step=20)
        assert loaded["metadata"]["step_num"] == 20
        assert loaded["metadata"]["value"] == 200
        print("   OK Specific checkpoint loaded correctly")

        # Load latest
        print("\n4. Loading latest checkpoint...")
        loaded_latest = load_checkpoint(checkpoint_dir)
        assert loaded_latest["metadata"]["step_num"] == 30
        print("   OK Latest checkpoint loaded correctly")

    print("\n" + "=" * 60)
    print("Test 2 PASSED")
    print("=" * 60)


def test_resume_simulation():
    """Simulate interruption and resume."""
    print("\n" + "=" * 60)
    print("Test 3: Resume Simulation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True)

        # Simulate first run
        print("\n1. Simulating first run (process chunks 1-10)...")
        chunk_count = 0
        total_loss = 0.0
        results = {"chunks": []}

        params = {"weights": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        opt = optax.adam(1e-3)
        opt_state = opt.init(params)

        for i in range(1, 11):
            chunk_count = i
            total_loss += i * 0.1
            results["chunks"].append({"id": i, "loss": i * 0.1})

        # Save checkpoint at chunk 10
        print("   Saving checkpoint at chunk 10...")
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=10,
            state={"params": params, "opt_state": opt_state},
            metadata={
                "chunk_count": chunk_count,
                "total_loss": total_loss,
                "results": results,
            },
        )
        wait_for_checkpoints()  # Wait for async save to complete
        print("   OK Checkpoint saved (interrupted)")

        # Simulate resume
        print("\n2. Simulating resume...")
        latest = get_latest_checkpoint_step(checkpoint_dir)
        print(f"   Found checkpoint at step {latest}")

        checkpoint = load_checkpoint(checkpoint_dir, latest)

        # Restore state
        chunk_count_resumed = checkpoint["metadata"]["chunk_count"]
        total_loss_resumed = checkpoint["metadata"]["total_loss"]
        results_resumed = checkpoint["metadata"]["results"]

        print(f"   OK Resumed from chunk {chunk_count_resumed}")
        print(f"   OK Total loss so far: {total_loss_resumed}")
        print(f"   OK Chunks processed: {len(results_resumed['chunks'])}")

        # Verify resume state
        assert chunk_count_resumed == 10
        assert len(results_resumed["chunks"]) == 10
        assert abs(total_loss_resumed - sum(i * 0.1 for i in range(1, 11))) < 1e-6
        print("   OK Resume state verified")

        # Continue from where we left off
        print("\n3. Continuing processing (chunks 11-15)...")
        for i in range(11, 16):
            chunk_count_resumed = i
            total_loss_resumed += i * 0.1
            results_resumed["chunks"].append({"id": i, "loss": i * 0.1})

        print(f"   OK Processed up to chunk {chunk_count_resumed}")
        print(f"   OK Total chunks: {len(results_resumed['chunks'])}")

        # Verify final state
        assert chunk_count_resumed == 15
        assert len(results_resumed["chunks"]) == 15
        print("   OK Final state verified")

    print("\n" + "=" * 60)
    print("Test 3 PASSED")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "[CHECK] Checkpointing Verification Tests")
    print("\n")

    try:
        test_basic_save_load()
        test_multiple_checkpoints()
        test_resume_simulation()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nCheckpointing implementation is working correctly!")

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
