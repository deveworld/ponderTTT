"""
Checkpointing utilities using Orbax with multi-host support and async saving.
"""

import threading
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp

# Global list to track async save threads
_save_threads = []
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

    def _save_thread():
        """Background thread that performs the actual save."""
        checkpointer = ocp.PyTreeCheckpointer()

        # Prepare checkpoint
        checkpoint = {
            "state": state,
            "step": step,
        }

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save (this blocks in the background thread, not main thread)
        checkpointer.save(
            checkpoint_dir / f"checkpoint_{step}",
            checkpoint,
            force=True,
        )

    # Start save in background thread
    thread = threading.Thread(target=_save_thread, daemon=False)
    thread.start()

    with _save_lock:
        _save_threads.append(thread)

    if process_index == 0:
        print(f" Checkpoint saved at step {step}")


def wait_for_checkpoints():
    """Wait for all pending checkpoint saves to complete."""
    global _save_threads
    with _save_lock:
        threads_to_wait = list(_save_threads)

    for thread in threads_to_wait:
        thread.join()

    with _save_lock:
        _save_threads.clear()


def finalize_checkpointing():
    """Finalize all checkpointing operations and cleanup."""
    wait_for_checkpoints()


def load_checkpoint(
    checkpoint_dir: str | Path,
    step: int | None = None,
) -> dict[str, Any]:
    """
    Load checkpoint using Orbax.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Specific step to load (if None, load latest)

    Returns:
        Checkpoint dictionary
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    checkpointer = ocp.PyTreeCheckpointer()

    if step is None:
        # Find latest checkpoint (exclude temporary files)
        checkpoints = [
            cp for cp in checkpoint_dir.glob("checkpoint_*")
            if not cp.name.endswith(".orbax-checkpoint-tmp")
        ]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("_")[1]))
        checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_{step}"

    # Load
    checkpoint = checkpointer.restore(checkpoint_path)

    return checkpoint


def get_latest_checkpoint_step(checkpoint_dir: str | Path) -> int | None:
    """
    Get the step number of the latest checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Latest step number, or None if no checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()

    # Exclude temporary files
    checkpoints = [
        cp for cp in checkpoint_dir.glob("checkpoint_*")
        if not cp.name.endswith(".orbax-checkpoint-tmp")
    ]
    if not checkpoints:
        return None

    # Extract step numbers
    steps = [int(cp.name.split("_")[1]) for cp in checkpoints]
    return max(steps)
