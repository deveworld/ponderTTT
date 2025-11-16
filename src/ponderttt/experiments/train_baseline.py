"""
Train baseline TTT models with fixed action schedules.

Usage:
    python -m ponderttt.experiments.train_baseline --model_scale 125m --action UPDATE_1
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import TTTConfig, TTTLayer
from ..training import TTTTrainer
from ..utils import init_rng, next_rng
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
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing (no dataset required)",
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
        "UPDATE_2": 5.0,
        "UPDATE_4": 12.0,
    }
    return mapping[action]


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

    # Create data iterator
    print("Creating data iterator...")
    if args.use_synthetic:
        print("Using synthetic data...")
        # Create synthetic data generator with VARIED tokens
        def synthetic_data_iter():
            num_chunks_per_seq = config.model.max_seq_length // config.model.chunk_size
            batch_idx = 0
            while True:
                # Generate random tokens from vocabulary (realistic)
                rng = jax.random.PRNGKey(config.seed + batch_idx)
                input_ids = jax.random.randint(
                    rng,
                    (config.training.batch_size, config.model.max_seq_length),
                    0,
                    min(10000, tokenizer.vocab_size),  # Use subset of vocab for faster generation
                    dtype=jnp.int32
                )

                # Create chunks from sequence
                chunks = input_ids.reshape(
                    config.training.batch_size,
                    num_chunks_per_seq,
                    config.model.chunk_size
                )

                yield {
                    "input_ids": input_ids,
                    "attention_mask": jnp.ones_like(input_ids),
                    "chunks": chunks,
                }
                batch_idx += 1
        data_iter = synthetic_data_iter()
    else:
        data_iter = create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            batch_size=config.training.batch_size,
            seq_length=config.model.max_seq_length,
            chunk_size=config.model.chunk_size,
            max_examples=config.training.num_train_examples,
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
                params=variables['params'],
                dropout_rng=dropout_rng,
                train=True
            )
        else:
            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                params=variables['params'],
                train=False
            )
        return {'logits': outputs.logits}

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

    from flax.training import train_state
    state = train_state.TrainState.create(
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

    with tqdm(total=args.max_chunks, desc="Processing chunks") as pbar:
        for _batch_idx, batch in enumerate(data_iter):
            if chunk_count >= args.max_chunks:
                break

            # Get chunks from batch
            chunks = batch["chunks"]  # [batch, num_chunks, chunk_size]
            batch_size, num_chunks, chunk_size = chunks.shape

            for i in range(num_chunks):
                if chunk_count >= args.max_chunks:
                    break

                chunk = chunks[:, i, :]  # [batch, chunk_size]

                # Prepare batch for training
                chunk_batch = {
                    'input_ids': chunk,
                    'attention_mask': jnp.ones_like(chunk),
                }

                # Compute actual loss using the model
                if num_steps > 0:
                    # Perform TTT updates
                    for _ in range(num_steps):
                        state, metrics = trainer.train_step(state, chunk_batch)
                    chunk_loss = metrics['loss']
                else:
                    # SKIP: just evaluate without training
                    metrics = trainer.eval_step(state, chunk_batch)
                    chunk_loss = metrics['loss']

                # Accumulate cost
                chunk_cost = action_to_cost(args.action)
                total_cost += chunk_cost
                total_loss += float(chunk_loss)
                chunk_count += 1

                results["chunks"].append({
                    "chunk_id": chunk_count,
                    "loss": float(chunk_loss),
                    "cost": chunk_cost,
                    "action": args.action,
                })

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{chunk_loss:.4f}",
                    "avg_cost": f"{total_cost/chunk_count:.2f}×",
                })

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
