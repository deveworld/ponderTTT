"""
Train baseline TTT models with fixed action schedules (NNX version).

Implements the architecture from PLAN.md:
- Slow weights (theta_slow): Frozen pretrained model
- Fast weights (theta_fast): Adaptive TTT layer weights

Usage:
    python -m ponderttt.experiments.train_baseline --model_scale 125m --action UPDATE_1
"""

import argparse
import json
import math
from pathlib import Path
from typing import cast

import jax
import optax
from flax import nnx
from tokenizers import Tokenizer
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model
from .training_utils import run_chunk_step


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline TTT model (NNX)")

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
        required=True,
        help="Fixed action to use throughout training",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=100, help="Maximum number of chunks to process"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baselines_nnx",
        help="Output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        default=True,
        help="Load pretrained GPT-2 weights (default: True, use --no-load-pretrained to disable)"
    )
    parser.add_argument(
        "--no-load-pretrained",
        action="store_false",
        dest="load_pretrained",
        help="Don't load pretrained weights (use random initialization)"
    )
    parser.add_argument(
        "--fast_weight_type",
        type=str,
        choices=["ttt", "lora"],
        default="ttt",
        help="Type of fast weights: 'ttt' (TTT Layer) or 'lora' (LoRA)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (only used if fast_weight_type='lora')"
    )

    return parser.parse_args()


def action_to_steps(action: str) -> int:
    """Convert action name to number of TTT steps."""
    mapping = {"SKIP": 0, "UPDATE_1": 1, "UPDATE_2": 2, "UPDATE_4": 4}
    return mapping[action]


def action_to_cost(action: str) -> float:
    """Convert action name to computational cost multiplier."""
    mapping = {"SKIP": 1.0, "UPDATE_1": 3.0, "UPDATE_2": 6.0, "UPDATE_4": 12.0}
    return mapping[action]


def get_model_name(model_scale: str) -> str:
    """Convert model scale to HuggingFace model name."""
    mapping = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
    }
    return mapping[model_scale]


def count_params(model: nnx.Module) -> int:
    """Count total parameters in NNX model."""
    state = nnx.state(model)
    return sum(x.size for x in jax.tree.leaves(state))


def main():
    args = parse_args()

    print("=" * 60)
    print("PonderTTT Baseline Training (NNX)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Action: {args.action}")
    print(f"Max chunks: {args.max_chunks}")
    print(f"Output dir: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model configuration
    model_name = get_model_name(args.model_scale)
    num_ttt_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)
    use_ttt = (num_ttt_steps > 0)

    print("\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  TTT steps: {num_ttt_steps}")
    print(f"  Cost multiplier: {cost_multiplier}x")
    print(f"  Use TTT: {use_ttt}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = cast(Tokenizer, get_tokenizer(model_name))

    # Create data iterator
    print("Creating data iterator...")
    batch_size = 4  # Small batch for CPU/testing
    seq_length = 2048
    chunk_size = 512

    chunks_per_sequence = seq_length // chunk_size
    examples_needed = math.ceil(args.max_chunks / max(chunks_per_sequence, 1))

    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=examples_needed * batch_size,
    )

    # Initialize model
    print(f"\nInitializing model with {args.fast_weight_type.upper()} fast weights...")

    # Prepare config based on fast weight type
    if args.fast_weight_type == "lora":
        from ponderttt.models import LoRAConfig
        lora_config = LoRAConfig(
            hidden_dim=768 if args.model_scale == "125m" else 1024 if args.model_scale == "350m" else 1280,
            rank=args.lora_rank,
            alpha=float(args.lora_rank),
            dropout_rate=0.1,
        )
        print(f"  LoRA rank: {args.lora_rank}")
        model, config = load_ttt_model(
            model_name=model_name,
            fast_weight_type="lora",
            lora_config=lora_config,
            seed=args.seed,
            load_pretrained=args.load_pretrained
        )
    else:
        model, config = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=args.load_pretrained
        )

    # Set to training mode
    model.train()

    print(f"OK Model loaded: {config.n_layer} layers, {config.n_embd} dim")
    print(f"  Fast weight type: {args.fast_weight_type}")
    total_params = count_params(model)
    print(f"  Total parameters: {total_params:,}")

    # Create optimizer (only optimize TTT layer parameters)
    print("\nCreating optimizer...")

    # Extract TTT layer parameters only (freeze slow weights)
    # Following PLAN.md: theta_slow (frozen), theta_fast (trainable)
    trainable_params = model.get_trainable_params()
    trainable_param_count = sum(x.size for x in jax.tree.leaves(trainable_params))

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters (TTT layer only): {trainable_param_count:,}")
    print(f"  Frozen parameters (base model): {total_params - trainable_param_count:,}")

    # Create optimizer for all parameters
    # Base model will be frozen via stop_gradient in the model's forward pass
    optimizer = nnx.Optimizer(model, optax.adam(args.learning_rate), wrt=nnx.All(nnx.Param))
    print(f"OK Optimizer: Adam (lr={args.learning_rate}, base_model frozen via stop_gradient)")

    # Training loop
    print("\nStarting training...")
    print(f"Processing {args.max_chunks} chunks...")

    total_loss = 0.0
    total_perplexity = 0.0
    total_cost = 0.0
    chunks_processed = 0

    action_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)

    with tqdm(total=args.max_chunks, desc="Training") as pbar:
        while chunks_processed < args.max_chunks:
            try:
                batch = next(data_iter)
            except StopIteration:
                print("\nData iterator exhausted")
                break

            num_chunks_available = batch["chunks"].shape[1]
            for chunk_idx in range(num_chunks_available):
                if chunks_processed >= args.max_chunks:
                    break

                chunk_batch = {
                    "input_ids": batch["chunks"][:, chunk_idx, :],
                    "attention_mask": batch["chunk_attention_mask"][:, chunk_idx, :],
                }

                if action_steps == 0:
                    metrics = run_chunk_step(
                        model,
                        None,
                        chunk_batch,
                        use_ttt=False,
                        apply_update=False,
                    )
                else:
                    metrics = None
                    for _ in range(action_steps):
                        metrics = run_chunk_step(
                            model,
                            optimizer,
                            chunk_batch,
                            use_ttt=True,
                            apply_update=True,
                        )

                assert metrics is not None
                total_loss += metrics["loss"]
                total_perplexity += metrics["perplexity"]
                total_cost += cost_multiplier
                chunks_processed += 1

                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss']:.4f}",
                        "ppl": f"{metrics['perplexity']:.2f}",
                    }
                )
                pbar.update(1)

                if chunks_processed % 10 == 0:
                    avg_loss = total_loss / chunks_processed
                    avg_ppl = total_perplexity / chunks_processed
                    print(f"\nChunk {chunks_processed}/{args.max_chunks}:")
                    print(f"  Average loss: {avg_loss:.4f}")
                    print(f"  Average perplexity: {avg_ppl:.2f}")
                    ttt_keys = [k for k in metrics.keys() if k.startswith("ttt_")]
                    if ttt_keys:
                        print("  TTT stats:")
                        for key in ttt_keys:
                            print(f"    {key}: {metrics[key]:.4f}")

    # Final results
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    if chunks_processed > 0:
        final_avg_loss = total_loss / chunks_processed
        final_avg_ppl = total_perplexity / chunks_processed

        print("\nFinal metrics:")
        print(f"  Chunks processed: {chunks_processed}")
        print(f"  Average loss: {final_avg_loss:.4f}")
        print(f"  Average perplexity: {final_avg_ppl:.2f}")
        print(f"  Average cost multiplier: {total_cost / max(chunks_processed, 1):.2f}x")

        # Save results
        results = {
            'model_scale': args.model_scale,
            'action': args.action,
            'fast_weight_type': args.fast_weight_type,
            'lora_rank': args.lora_rank if args.fast_weight_type == 'lora' else None,
            'num_ttt_steps': num_ttt_steps,
            'cost_multiplier': cost_multiplier,
            'chunks_processed': chunks_processed,
            'final_loss': final_avg_loss,
            'final_perplexity': final_avg_ppl,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'total_cost': total_cost,
            'avg_cost_per_chunk': total_cost / chunks_processed,
        }

        # Include fast_weight_type in filename
        suffix = f"_{args.fast_weight_type}"
        if args.fast_weight_type == "lora":
            suffix += f"_r{args.lora_rank}"
        results_file = output_dir / f"results_{args.model_scale}_{args.action}{suffix}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nOK Results saved to: {results_file}")
    else:
        print("\nNo chunks processed!")


if __name__ == "__main__":
    main()
