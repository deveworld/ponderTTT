"""
Checkpointing utilities using Orbax with multi-host support and async saving.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp

# Global list to track async save threads
_save_threads = []
_save_errors: list[Exception] = []
_save_lock = threading.Lock()


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    state: Any,
    metadata: dict | None = None,
    save_on_all_hosts: bool = False,
):
    """
    Save checkpoint asynchronously using background thread.

    Args:
        checkpoint_dir: Directory to save checkpoint
        step: Training step
        state: Training state to save
        metadata: Optional metadata
        save_on_all_hosts: If False, only host 0 saves (default for replicated state)
                          If True, each host saves its shard (for FSDP)

    Note:
        For replicated parameters (data parallel), use save_on_all_hosts=False
        For sharded parameters (FSDP), use save_on_all_hosts=True

        Saves are asynchronous - a background thread handles the I/O.
        Call wait_for_checkpoints() to ensure all saves complete.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    # In multi-host setup, coordinate saves
    try:
        process_index = jax.process_index()
    except (RuntimeError, ValueError):
        # JAX distributed not initialized (single host)
        process_index = 0

    # Only proceed if we should save from this host
    if not save_on_all_hosts and process_index != 0:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Make deep copies of data to avoid race conditions
    # (data might change in main thread while background thread is saving)
    state_copy = jax.tree_util.tree_map(
        lambda x: x.copy() if hasattr(x, "copy") else x,
        state,
    )
    metadata_copy = copy.deepcopy(metadata) if metadata is not None else None

    def _save_thread():
        """Background thread that performs the actual save."""
        try:
            checkpointer = ocp.PyTreeCheckpointer()

            # Prepare checkpoint using the copied data
            checkpoint = {
                "state": state_copy,
                "step": step,
            }

            if metadata_copy is not None:
                checkpoint["metadata"] = metadata_copy

            # Save (this blocks in the background thread, not main thread)
            checkpointer.save(
                checkpoint_dir / f"checkpoint_{step}",
                checkpoint,
                force=True,
            )
        except Exception as e:  # pragma: no cover - surfaced in wait_for_checkpoints
            with _save_lock:
                _save_errors.append(e)

    # Start save in background thread
    thread = threading.Thread(target=_save_thread, daemon=False)
    thread.start()

    with _save_lock:
        _save_threads.append(thread)

    if process_index == 0:
        print(f" Checkpoint saved at step {step}")


def get_latest_checkpoint_step(checkpoint_dir: str | Path) -> int | None:
    """Get the step number of the latest checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Latest checkpoint step, or None if no checkpoints exist
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    # Find all checkpoint directories (exclude temporary files)
    checkpoints = [
        cp
        for cp in checkpoint_dir.glob("checkpoint_*")
        if not cp.name.endswith(".orbax-checkpoint-tmp") and cp.is_dir()
    ]

    if not checkpoints:
        return None

    # Extract step numbers and return the maximum
    steps = []
    for cp in checkpoints:
        try:
            step = int(cp.name.split("_")[1])
            steps.append(step)
        except (IndexError, ValueError):
            continue

    return max(steps) if steps else None


def wait_for_checkpoints():
    """Wait for all pending checkpoint saves to complete."""
    global _save_threads
    with _save_lock:
        threads_to_wait = list(_save_threads)

    for thread in threads_to_wait:
        thread.join()

    with _save_lock:
        _save_threads.clear()
        if _save_errors:
            errors, _save_errors[:] = list(_save_errors), []
            raise RuntimeError(f"Checkpointing failed with errors: {errors}")


def load_checkpoint(
    checkpoint_dir: str | Path,
    step: int | None = None,
    target: Any | None = None,
) -> dict[str, Any]:
    """
    Load checkpoint using Orbax.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (if None, load latest)
        target: Optional target structure for restoration (e.g., for NNX State)

    Returns:
        Checkpoint dictionary
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    checkpointer = ocp.PyTreeCheckpointer()

    if step is None:
        # Find latest checkpoint (exclude temporary files)
        checkpoints = [
            cp
            for cp in checkpoint_dir.glob("checkpoint_*")
            if not cp.name.endswith(".orbax-checkpoint-tmp") and cp.is_dir()
        ]

        if not checkpoints:
            # Fallback: check if the directory itself is a checkpoint
            checkpoint_path = checkpoint_dir
        else:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.name.split("_")[1]))
            checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}"

    # Load with fallback to CPU sharding for cross-device compatibility
    def _try_restore():
        if target is not None:
            return checkpointer.restore(checkpoint_path, item=target)
        else:
            return checkpointer.restore(checkpoint_path)

    try:
        checkpoint = _try_restore()
    except ValueError as e:
        if "sharding" in str(e).lower():
            # Checkpoint was saved on GPU but we're restoring on CPU
            # Use ocp.args.Composite with CPU sharding for all arrays
            print("Cross-device restore: falling back to CPU sharding...")

            cpu_device = jax.devices("cpu")[0]
            cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)

            # Read metadata to get the tree structure
            metadata_path = checkpoint_path / "default" / "_METADATA"
            if not metadata_path.exists():
                # Try alternative metadata location
                metadata_path = checkpoint_path / "_METADATA"

            if metadata_path.exists():
                import json

                with open(metadata_path, "r") as f:
                    ckpt_metadata = json.load(f)

                # Build restore_args tree matching checkpoint structure
                from orbax.checkpoint import ArrayRestoreArgs

                def build_restore_args(tree):
                    """Recursively build restore args with CPU sharding."""
                    if isinstance(tree, dict):
                        return {k: build_restore_args(v) for k, v in tree.items()}
                    elif isinstance(tree, list):
                        return [build_restore_args(v) for v in tree]
                    else:
                        # Leaf node - use CPU sharding
                        return ArrayRestoreArgs(sharding=cpu_sharding)

                # Extract tree structure from metadata
                tree_metadata = ckpt_metadata.get("tree_metadata", ckpt_metadata)
                restore_args_tree = build_restore_args(tree_metadata)

                # Try with composite args
                args = ocp.args.Composite(
                    default=ocp.args.PyTreeRestore(restore_args=restore_args_tree)
                )
                checkpoint = checkpointer.restore(checkpoint_path, args=args)
            else:
                raise ValueError(f"Cannot find metadata at {metadata_path}: {e}")
        else:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    except Exception as e:
        if "sharding" in str(e).lower():
            raise ValueError(
                f"Failed to load checkpoint from {checkpoint_path}: {e}\n"
                "Hint: This checkpoint was saved on a different device (e.g., CUDA). "
                "Try loading on the same device type or use export_checkpoint.py to convert it."
            )
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    return checkpoint


def unwrap_state(state: Any) -> Any:
    """Recursively unwrap Orbax-serialized NNX state dicts (remove 'value' wrappers).

    Also converts string keys that look like integers back to integers,
    since Orbax may serialize integer keys as strings but NNX List needs int indices.

    Args:
        state: Orbax checkpoint state (possibly nested dict with 'value' wrappers)

    Returns:
        Unwrapped state suitable for NNX update operations
    """
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        # Convert string keys that are integers back to int for NNX List
        result = {}
        for k, v in state.items():
            # Try to convert string keys that look like integers
            if isinstance(k, str) and k.isdigit():
                k = int(k)
            result[k] = unwrap_state(v)
        return result
    return state
