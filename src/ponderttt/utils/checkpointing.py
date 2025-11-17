"""
Checkpointing utilities using Orbax with multi-host support.
"""

from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    state: Any,
    metadata: dict | None = None,
    save_on_all_hosts: bool = False,
):
    """
    Save checkpoint using Orbax with multi-host support.

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
    """
    checkpoint_dir = Path(checkpoint_dir)

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

    checkpointer = ocp.PyTreeCheckpointer()

    # Prepare checkpoint
    checkpoint = {
        "state": state,
        "step": step,
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Save
    checkpointer.save(
        checkpoint_dir / f"checkpoint_{step}",
        checkpoint,
    )

    if process_index == 0:
        print(f" Checkpoint saved at step {step}")


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
    checkpoint_dir = Path(checkpoint_dir)

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
    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None

    # Extract step numbers
    steps = [int(cp.name.split("_")[1]) for cp in checkpoints]
    return max(steps)
