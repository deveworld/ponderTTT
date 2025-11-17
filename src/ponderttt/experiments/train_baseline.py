"""
Train baseline TTT models with fixed action schedules.

Implements the architecture from PLAN.md:
- Slow weights (θ_slow): Frozen pretrained model
- Fast weights (θ_fast): Adaptive TTT layer weights

Usage:
    python -m ponderttt.experiments.train_baseline_ttt --model_scale 125m --action UPDATE_1
"""

import argparse
import json
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..data import create_data_iterator, get_tokenizer
from ..models import TTTConfig, load_ttt_model
from ..utils import init_rng, next_rng
from ..utils.checkpointing import (
    finalize_checkpointing,
    get_latest_checkpoint_step,
    load_checkpoint,
    save_checkpoint,
)
from .config import get_1b_config, get_125m_config, get_350m_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline TTT model")

    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["125m", "350m", "1b"],
        default="125m",
        help="Model scale",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"],
        default="UPDATE_1",
        help="Fixed action to use for all chunks",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=100,
        help="Maximum number of chunks to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baselines_ttt",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save checkpoint every N chunks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if available",
    )

    return parser.parse_args()


def action_to_steps(action: str) -> int:
    """Convert action name to number of TTT steps."""
    mapping = {
        "SKIP": 0,
        "UPDATE_1": 1,
        "UPDATE_2": 2,
        "UPDATE_4": 4,
    }
    return mapping[action]


def action_to_cost(action: str) -> float:
    """Convert action name to computational cost."""
    mapping = {
        "SKIP": 1.0,
        "UPDATE_1": 3.0,
        "UPDATE_2": 6.0,
        "UPDATE_4": 12.0,
    }
    return mapping[action]


def get_checkpoint_dir(output_dir: str, model_scale: str, action: str) -> Path:
    """Get checkpoint directory path."""
    return Path(output_dir).resolve() / model_scale / action / "checkpoints"


def try_load_checkpoint(checkpoint_dir: Path):
    """Try to load latest checkpoint. Returns None if no checkpoint exists."""
    try:
        if not checkpoint_dir.exists():
            return None

        latest_step = get_latest_checkpoint_step(checkpoint_dir)
        if latest_step is None:
            return None

        print(f"\nFound checkpoint at step {latest_step}")
        checkpoint = load_checkpoint(checkpoint_dir, latest_step)
        print(f"Successfully loaded checkpoint from step {latest_step}")
        return checkpoint

    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Baseline Training (TTT Architecture)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Action: {args.action}")
    print(f"Max chunks: {args.max_chunks}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Get configuration
    if args.model_scale == "125m":
        config = get_125m_config()
    elif args.model_scale == "350m":
        config = get_350m_config()
    else:
        config = get_1b_config()

    config.seed = args.seed
    config.output_dir = args.output_dir

    # Initialize RNG
    init_rng(config.seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = cast(PreTrainedTokenizer, get_tokenizer(config.model.model_name))

    # Create data iterator from The Stack dataset
    print("Creating data iterator...")
    import math

    num_chunks_per_seq = config.model.max_seq_length // config.model.chunk_size
    num_batches_needed = math.ceil(
        args.max_chunks / (num_chunks_per_seq * config.training.batch_size)
    )
    num_examples_needed = int(num_batches_needed * config.training.batch_size * 1.5)

    print(f"Downloading {num_examples_needed} examples for {args.max_chunks} chunks...")

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=config.training.batch_size,
        seq_length=config.model.max_seq_length,
        chunk_size=config.model.chunk_size,
        max_examples=num_examples_needed,
    )

    # Initialize TTT model
    print("Initializing TTT model...")
    ttt_config = TTTConfig(
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        ttt_hidden_dim=config.model.ttt_hidden_dim,
        chunk_size=config.model.chunk_size,
        max_seq_length=config.model.max_seq_length,
        dtype=jnp.float32,
    )

    model, _ = load_ttt_model(
        model_name=config.model.model_name,
        ttt_config=ttt_config,
        dtype=jnp.float32,
    )

    # Initialize model parameters
    rng_key = next_rng()
    rng_key = (
        jax.random.PRNGKey(config.seed)
        if not isinstance(rng_key, jax.Array)
        else rng_key
    )
    dummy_input = jnp.ones((1, config.model.max_seq_length), dtype=jnp.int32)
    variables = model.init(rng_key, dummy_input, deterministic=True)
    params = variables["params"]

    print("\nModel architecture:")
    print(f"  Slow weights: {config.model.model_name} (frozen)")
    print("  Fast weights: TTT layer (adaptive)")
    print(f"  TTT hidden dim: {config.model.ttt_hidden_dim}")
    print()

    # Create optimizer (only for TTT layer parameters)
    # Base model parameters are frozen
    optimizer = optax.adam(config.training.learning_rate)

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # Training loop
    print("\nStarting training...")
    print(f"Action: {args.action} ({action_to_steps(args.action)} TTT steps)")
    print(f"Cost: {action_to_cost(args.action)}×")
    print()

    num_steps = action_to_steps(args.action)
    total_cost = 0.0
    total_loss = 0.0
    chunk_count = 0

    results = {
        "config": {
            "model_scale": args.model_scale,
            "action": args.action,
            "num_steps": num_steps,
            "cost_per_chunk": action_to_cost(args.action),
            "architecture": "TTT (slow + fast weights)",
        },
        "chunks": [],
    }

    # Save base TTT parameters (will be restored for each batch)
    base_params = state.params

    # Checkpoint management
    checkpoint_dir = get_checkpoint_dir(args.output_dir, args.model_scale, args.action)

    # Try to resume from checkpoint
    if args.resume:
        checkpoint = try_load_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            # Restore state
            state = state.replace(
                params=checkpoint["state"]["params"],
                opt_state=checkpoint["state"]["opt_state"],
            )
            base_params = state.params

            # Restore progress
            chunk_count = checkpoint["metadata"]["chunk_count"]
            total_cost = checkpoint["metadata"]["total_cost"]
            total_loss = checkpoint["metadata"]["total_loss"]
            results = checkpoint["metadata"]["results"]

            print(f"Resumed from chunk {chunk_count}")
            print(f"Resuming with {args.max_chunks - chunk_count} chunks remaining")

            # Skip already-processed batches to maintain data consistency
            num_chunks_per_batch = num_chunks_per_seq * config.training.batch_size
            batches_to_skip = chunk_count // num_chunks_per_batch

            if batches_to_skip > 0:
                print(f"Skipping {batches_to_skip} already-processed batches...")
                for _ in range(batches_to_skip):
                    try:
                        next(data_iter)
                    except StopIteration:
                        print("Warning: Reached end of data while skipping batches")
                        break
                print(f"✓ Skipped {batches_to_skip} batches\n")
            else:
                print()

    # Define loss function for TTT updates
    def compute_loss(params, batch):
        """Compute language modeling loss with TTT layer."""
        outputs = model.apply(
            {"params": params},
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            deterministic=True,
            use_ttt=(num_steps > 0),  # Skip TTT for SKIP action
        )

        logits = outputs["logits"]
        labels = batch["input_ids"][:, 1:]
        logits = logits[:, :-1]

        vocab_size = logits.shape[-1]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
        )

        mask = batch["attention_mask"][:, 1:]
        loss = loss * mask.reshape(-1)
        loss = jnp.sum(loss) / jnp.sum(mask)

        return loss

    # JIT compile for speed
    @jax.jit
    def ttt_update_step(params, opt_state, batch):
        """Single TTT gradient step on fast weights."""
        loss, grads = jax.value_and_grad(compute_loss)(params, batch)

        # Update only TTT layer parameters (fast weights)
        # Base model parameters remain frozen
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    @jax.jit
    def evaluate(params, batch):
        """Evaluate with current parameters."""
        loss = compute_loss(params, batch)
        return loss

    with tqdm(
        total=args.max_chunks, initial=chunk_count, desc="Processing chunks"
    ) as pbar:
        for _batch_idx, batch in enumerate(data_iter):
            if chunk_count >= args.max_chunks:
                break

            # Start from base parameters for each batch (batch independence)
            batch_params = base_params
            batch_opt_state = state.opt_state

            if num_steps == 0:
                # SKIP: No TTT, just evaluate with frozen base model
                sequence_loss = evaluate(batch_params, batch)
            else:
                # UPDATE: Adapt TTT fast weights N times on full sequence
                current_params = batch_params
                current_opt_state = batch_opt_state
                for _ in range(num_steps):
                    current_params, current_opt_state, _ = ttt_update_step(
                        current_params, current_opt_state, batch
                    )

                # Evaluate with adapted fast weights
                sequence_loss = evaluate(current_params, batch)

            # Record results per chunk (for fair comparison)
            chunks = batch["chunks"]
            batch_sz, num_chunks, chunk_sz = chunks.shape

            # Process all sequences in the batch
            for seq_idx in range(batch_sz):
                for chunk_idx in range(num_chunks):
                    if chunk_count >= args.max_chunks:
                        break

                    chunk_cost = action_to_cost(args.action)
                    total_cost += chunk_cost
                    total_loss += float(sequence_loss)
                    chunk_count += 1

                    results["chunks"].append(
                        {
                            "chunk_id": chunk_count,
                            "sequence_idx": seq_idx,
                            "chunk_position": chunk_idx,
                            "loss": float(sequence_loss),
                            "cost": chunk_cost,
                            "action": args.action,
                        }
                    )

                    # Save checkpoint periodically
                    if chunk_count % args.checkpoint_every == 0:
                        try:
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            save_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                step=chunk_count,
                                state={
                                    "params": state.params,
                                    "opt_state": state.opt_state,
                                },
                                metadata={
                                    "chunk_count": chunk_count,
                                    "total_cost": total_cost,
                                    "total_loss": total_loss,
                                    "results": results,
                                },
                            )
                            pbar.write(f"Checkpoint saved at chunk {chunk_count}")
                        except Exception as e:
                            pbar.write(f"Warning: Checkpoint save failed: {e}")

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": f"{sequence_loss:.4f}",
                            "avg_cost": f"{total_cost / chunk_count:.2f}×",
                        }
                    )

                if chunk_count >= args.max_chunks:
                    break

    # Compute final statistics
    avg_loss = total_loss / chunk_count
    avg_cost = total_cost / chunk_count

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total chunks processed: {chunk_count}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average cost: {avg_cost:.2f}×")
    print(f"Total cost: {total_cost:.2f}×")

    # Save results
    output_dir = Path(args.output_dir) / args.model_scale
    output_dir.mkdir(parents=True, exist_ok=True)

    results["summary"] = {
        "total_chunks": chunk_count,
        "avg_loss": avg_loss,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
    }

    output_file = output_dir / f"{args.action}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Wait for any pending checkpoint saves to complete
    print("\nFinalizing checkpoints...")
    finalize_checkpointing()
    print("Done.")


if __name__ == "__main__":
    main()
