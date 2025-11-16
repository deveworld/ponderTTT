"""
Training script for TPU with multi-host support.

This script demonstrates proper JAX distributed setup for TPU Pods.

Usage:
    # Single host (8 TPU chips)
    python scripts/train_tpu.py --mesh_shape="8,1"

    # Multi-host (e.g., TPU v4-64: 8 hosts × 8 chips)
    # Run this on ALL hosts simultaneously
    python scripts/train_tpu.py --mesh_shape="64,1" --multi_host

Example on Google Cloud TPU v4-64:
    gcloud compute tpus tpu-vm ssh ponderttt-v4-64 \
        --zone=us-central2-b \
        --worker=all \
        --command="cd ~/ponderttt && python scripts/train_tpu.py --multi_host"
"""

import sys
from pathlib import Path
from typing import cast
from transformers import PreTrainedTokenizer


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import optax
from functools import partial
import argparse

from ponderttt.utils import (
    initialize_jax_distributed,
    create_mesh,
    shard_batch,
    get_local_batch_size,
    print_on_main,
    save_checkpoint,
    init_rng,
    next_rng,
    cross_entropy_loss,
)
from ponderttt.data import get_tokenizer, create_data_iterator
from ponderttt.models import TransformerLM, ModelConfig, inspect_sharding


def parse_args():
    parser = argparse.ArgumentParser(description="Train on TPU")

    # Distributed setup
    parser.add_argument(
        "--multi_host",
        action="store_true",
        help="Enable multi-host distributed training"
    )
    parser.add_argument(
        "--coordinator_address",
        type=str,
        default=None,
        help="Coordinator address for multi-host (auto-detected if None)"
    )
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="8,1",
        help="Mesh shape as 'batch,model' (e.g., '64,1' for data parallel)"
    )
    parser.add_argument(
        "--mesh_axes",
        type=str,
        default="batch,model",
        help="Mesh axis names (default: 'batch,model')"
    )

    # Training config
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()


def create_train_state(config, mesh, learning_rate, warmup_steps, num_steps):
    """Create initial training state."""
    # Initialize model
    model = TransformerLM(config=config)

    # Initialize parameters
    rng = next_rng()
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables['params']

    # Apply parameter sharding if configured
    if config.shard_params and config.mesh is not None:
        from ponderttt.models import apply_sharding_to_params
        print_on_main("Applying parameter sharding...")
        params = apply_sharding_to_params(params, config.mesh)
        print_on_main(" Parameters sharded across devices")

        # Inspect sharding (only shows first 20 params)
        inspect_sharding(params, max_params=20)

    # Create optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    return {
        'params': params,
        'opt_state': opt_state,
        'step': 0,
    }, optimizer, model


@partial(jax.jit, donate_argnums=(0,))
def train_step(state, batch, model, optimizer, mesh, use_sharding_constraint=True):
    """
    Single training step with sharding constraints for FSDP.

    In FSDP mode, gradients automatically inherit parameter sharding.
    JAX's autodiff ensures gradients have the same sharding as parameters,
    and the compiler inserts necessary Reduce-Scatter operations.

    Args:
        state: Training state
        batch: Input batch
        model: Flax model
        optimizer: Optax optimizer
        mesh: JAX mesh for sharding
        use_sharding_constraint: Whether to apply sharding constraints
    """
    from jax.lax import with_sharding_constraint
    from jax.sharding import NamedSharding, PartitionSpec as PS

    # Define sharding specs for data
    batch_sharding = NamedSharding(mesh, PS('batch', None))

    def loss_fn(params):
        # Constrain input sharding
        if use_sharding_constraint:
            input_ids = with_sharding_constraint(
                batch['input_ids'],
                batch_sharding
            )
        else:
            input_ids = batch['input_ids']

        # Forward pass
        logits = model.apply(
            {'params': params},
            input_ids,
        )['logits']

        # Constrain output sharding
        if use_sharding_constraint:
            logits = with_sharding_constraint(
                logits,
                NamedSharding(mesh, PS('batch', None, None))
            )

        # Compute loss (shift for next token prediction)
        loss = cross_entropy_loss(
            logits[:, :-1],
            batch['input_ids'][:, 1:],
            batch['attention_mask'][:, 1:],
        )
        return loss

    # Compute gradients
    # Gradients automatically inherit parameter sharding in FSDP
    # JAX's autodiff propagates sharding from parameters to gradients
    loss, grads = jax.value_and_grad(loss_fn)(state['params'])

    # Update parameters
    updates, new_opt_state = optimizer.update(
        grads,
        state['opt_state'],
        state['params'],
    )
    new_params = optax.apply_updates(state['params'], updates)

    # Update state
    new_state = {
        'params': new_params,
        'opt_state': new_opt_state,
        'step': state['step'] + 1,
    }

    # Metrics
    metrics = {
        'loss': loss,
        'step': state['step'],
    }

    return new_state, metrics


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Training on TPU")
    print("=" * 60)

    # Step 1: Initialize JAX distributed
    if args.multi_host:
        print("\n[1/6] Initializing multi-host JAX...")
        initialize_jax_distributed(
            coordinator_address=args.coordinator_address,
        )
    else:
        print("\n[1/6] Single-host mode")
        print(f"Local devices: {jax.local_device_count()}")

    # Step 2: Create mesh
    print("\n[2/6] Creating device mesh...")
    mesh_shape = tuple(map(int, args.mesh_shape.split(",")))
    mesh_axes = tuple(args.mesh_axes.split(","))
    mesh = create_mesh(mesh_shape, mesh_axes)

    # Step 3: Calculate batch sizes
    print("\n[3/6] Setting up data pipeline...")
    local_batch_size = get_local_batch_size(args.global_batch_size)
    print_on_main(f"Local batch size per host: {local_batch_size}")

    # Step 4: Load data
    tokenizer = get_tokenizer(args.model_name)
    tokenizer = cast(PreTrainedTokenizer, tokenizer)
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=local_batch_size,
        seq_length=args.seq_length,
        chunk_size=args.chunk_size,
    )

    # Step 5: Initialize model and optimizer
    print("\n[4/6] Initializing model and optimizer...")
    init_rng(42)

    # Enable parameter sharding for large models (1B+)
    # For smaller models (<1B), this can be set to False
    shard_params = True  # Recommended for TPU v4-64

    model_config = ModelConfig(
        model_name=args.model_name,
        mesh=mesh,
        shard_params=shard_params,
    )

    with mesh:
        state, optimizer, model = create_train_state(
            model_config,
            mesh,
            args.learning_rate,
            args.warmup_steps,
            args.num_steps,
        )

    print_on_main(f" Model initialized: {args.model_name}")
    print_on_main(" Optimizer: Adam with cosine decay")

    # Step 6: Training loop
    print("\n[5/6] Starting training...")
    print_on_main(f"Training for {args.num_steps} steps")
    print_on_main(f"Global batch size: {args.global_batch_size}")

    with mesh:
        for step in range(args.num_steps):
            # Get batch
            batch = next(data_iter)

            # Shard batch across devices
            batch = shard_batch(batch, mesh, batch_axis=mesh_axes[0])

            # Training step with sharding constraints
            state, metrics = train_step(
                state, batch, model, optimizer, mesh,
                use_sharding_constraint=True
            )

            # Logging
            if step % args.log_every == 0:
                loss = float(metrics['loss'])
                print_on_main(f"Step {step:5d} | Loss: {loss:.4f}")

            # Checkpointing
            if step > 0 and step % args.checkpoint_every == 0:
                print_on_main(f"\nSaving checkpoint at step {step}...")
                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    step=step,
                    state=state,
                    metadata={'args': vars(args)},
                )

    print("\n" + "=" * 60)
    print(" Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
