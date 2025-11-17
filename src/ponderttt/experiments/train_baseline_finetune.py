"""
Train baseline TTT models with fixed action schedules.

Usage:
    python -m ponderttt.experiments.train_baseline --model_scale 125m --action UPDATE_1
"""

import argparse
import copy
import json
from pathlib import Path
from typing import cast

import jax.numpy as jnp
from flax.core import freeze, unfreeze
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..data import create_data_iterator, get_tokenizer
from ..models import TTTConfig, TTTLayer
from ..training import TTTTrainer, TrainState
from ..utils import init_rng, next_rng
from ..utils.checkpointing import (
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
        default="outputs/baselines",
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
    return Path(output_dir) / model_scale / action / "checkpoints"


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
    print("PonderTTT Baseline Training")
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
    tokenizer = get_tokenizer(config.model.model_name)
    tokenizer = cast(PreTrainedTokenizer, tokenizer)

    # Create data iterator from The Stack dataset
    print("Creating data iterator...")
    import math

    num_chunks_per_seq = config.model.max_seq_length // config.model.chunk_size
    num_batches_needed = math.ceil(args.max_chunks / num_chunks_per_seq)
    num_examples_needed = int(num_batches_needed * config.training.batch_size * 1.5)

    print(
        f"Downloading {num_examples_needed} examples for {args.max_chunks} chunks ({num_batches_needed} batches)..."
    )

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=config.training.batch_size,
        seq_length=config.model.max_seq_length,
        chunk_size=config.model.chunk_size,
        max_examples=num_examples_needed,
    )

    # Initialize model
    print("Initializing model...")
    from transformers import FlaxAutoModelForCausalLM

    hf_model = FlaxAutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        dtype=jnp.float32,
    )
    params = hf_model.params

    # Wrapper to make HF model compatible with trainer
    def model_apply_fn(variables, input_ids, attention_mask=None, deterministic=True):
        # HF models need dropout RNG when train=True
        if not deterministic:
            dropout_rng = next_rng()
            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=variables["params"],
                dropout_rng=dropout_rng,
                train=True,
            )
        else:
            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=variables["params"],
                train=False,
            )
        return {"logits": outputs.logits}

    # Create a simple model object
    class ModelWrapper:
        def __init__(self, apply_fn):
            self.apply = apply_fn
            self.__call__ = apply_fn

    model = ModelWrapper(model_apply_fn)

    # Initialize TTT layer
    ttt_config = TTTConfig(
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        head_dim=config.model.head_dim,
        ttt_hidden_dim=config.model.ttt_hidden_dim,
        chunk_size=config.model.chunk_size,
    )
    TTTLayer(config=ttt_config)

    # Create trainer with model (Note: using HF model directly)
    trainer = TTTTrainer(model=model)

    import optax

    optimizer = optax.adam(config.training.learning_rate)
    optimizer.init(params)

    state = TrainState.create(
        apply_fn=model_apply_fn,  # Use wrapper function
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
        },
        "chunks": [],
    }

    # Save base model state (will be restored for each batch)
    base_params = copy.deepcopy(unfreeze(state.params))

    # Checkpoint management
    checkpoint_dir = get_checkpoint_dir(args.output_dir, args.model_scale, args.action)

    # Try to resume from checkpoint
    if args.resume:
        checkpoint = try_load_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            # Restore state
            state = state.replace(
                params=freeze(checkpoint["state"]["params"]),
                opt_state=checkpoint["state"]["opt_state"],
            )
            base_params = copy.deepcopy(unfreeze(state.params))

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

    with tqdm(
        total=args.max_chunks, initial=chunk_count, desc="Processing chunks"
    ) as pbar:
        for _batch_idx, batch in enumerate(data_iter):
            if chunk_count >= args.max_chunks:
                break

            if num_steps == 0:
                # SKIP: Evaluate full sequence at once (proper context)
                full_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                metrics = trainer.eval_step(state, full_batch)
                sequence_loss = metrics["loss"]

                # Count as num_chunks for fair comparison
                chunks = batch["chunks"]
                batch_size, num_chunks, chunk_size = chunks.shape

                for i in range(num_chunks):
                    if chunk_count >= args.max_chunks:
                        break

                    chunk_cost = action_to_cost(args.action)
                    total_cost += chunk_cost
                    total_loss += float(sequence_loss)
                    chunk_count += 1

                    results["chunks"].append(
                        {
                            "chunk_id": chunk_count,
                            "chunk_position": i,
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
                                    "params": unfreeze(state.params),
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
            else:
                # UPDATE: Adapt on full sequences
                # Start from base model for each batch
                batch_state = state.replace(params=freeze(copy.deepcopy(base_params)))

                full_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                # Adapt model on FULL sequence
                for _ in range(num_steps):
                    batch_state, _ = trainer.train_step(batch_state, full_batch)

                # Evaluate full sequence to get proper loss
                metrics = trainer.eval_step(batch_state, full_batch)
                sequence_loss = metrics["loss"]

                # Count cost per chunk for fair comparison
                chunks = batch["chunks"]
                batch_size, num_chunks, chunk_size = chunks.shape

                for i in range(num_chunks):
                    if chunk_count >= args.max_chunks:
                        break

                    chunk_cost = action_to_cost(args.action)
                    total_cost += chunk_cost
                    total_loss += float(sequence_loss)
                    chunk_count += 1

                    results["chunks"].append(
                        {
                            "chunk_id": chunk_count,
                            "chunk_position": i,
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
                                    "params": unfreeze(state.params),
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


if __name__ == "__main__":
    main()
