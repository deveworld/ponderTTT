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

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from typing import cast

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTTransformerLM
from ..models.gpt2_nnx import GPT2Config
from ..utils.checkpointing import save_checkpoint, wait_for_checkpoints, load_checkpoint
from .training_utils import run_chunk_step
import wandb


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
    parser.add_argument(
        "--ssl_weight",
        type=float,
        default=0.1,
        help="Weight for SSL auxiliary loss when using TTT/LoRA fast weights",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated list of seeds for multi-seed runs (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (if None, WandB is disabled)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N chunks",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for data downloading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )

    return parser.parse_args()


def action_to_steps(action: str) -> int:
    """Convert action name to number of TTT steps."""
    mapping = {"SKIP": 0, "UPDATE_1": 1, "UPDATE_2": 2, "UPDATE_4": 4}
    return mapping[action]


def action_to_cost(action: str) -> float:
    """Convert action name to computational cost multiplier.

    Cost model: 1 (base forward) + 2 * num_steps (backward + update per step)
    - SKIP: 1 + 2*0 = 1.0
    - UPDATE_1: 1 + 2*1 = 3.0
    - UPDATE_2: 1 + 2*2 = 5.0
    - UPDATE_4: 1 + 2*4 = 9.0
    """
    mapping = {"SKIP": 1.0, "UPDATE_1": 3.0, "UPDATE_2": 5.0, "UPDATE_4": 9.0}
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
    seeds = [args.seed] if args.seeds is None else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 60)
    print("PonderTTT Baseline Training (NNX)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Action: {args.action}")
    print(f"Max chunks: {args.max_chunks}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seeds: {seeds}")

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

    # Initialize WandB (Global)
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"baseline_{args.model_scale}_{args.action}",
        )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(model_name)

    # Create data iterator
    print("Creating data iterator...")
    batch_size = args.batch_size
    seq_length = 1024
    chunk_size = 512

    chunks_per_sequence = seq_length // chunk_size
    # Each batch processes batch_size * chunks_per_sequence chunks
    chunks_per_batch = batch_size * chunks_per_sequence
    batches_needed = math.ceil(args.max_chunks / chunks_per_batch)
    examples_needed = batches_needed * batch_size  # sequences needed

    def build_data_iter():
        return create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            batch_size=batch_size,
            seq_length=seq_length,
            chunk_size=chunk_size,
            max_examples=examples_needed,
            num_workers=args.num_workers,
        )

    def init_model(seed):
        print(f"\nInitializing model with {args.fast_weight_type.upper()} fast weights (seed={seed})...")
        tok_vocab_size = tokenizer.get_vocab_size()
        print(f"  Tokenizer vocab_size: {tok_vocab_size}")

        if args.fast_weight_type == "lora":
            from ponderttt.models import LoRAConfig
            lora_config = LoRAConfig(
                hidden_dim=768 if args.model_scale == "125m" else 1024 if args.model_scale == "350m" else 1280,
                rank=args.lora_rank,
                alpha=float(args.lora_rank),
                dropout_rate=0.1,
            )
            print(f"  LoRA rank: {args.lora_rank}")
            mdl_raw, cfg_raw = load_ttt_model(
                model_name=model_name,
                fast_weight_type="lora",
                lora_config=lora_config,
                seed=seed,
                load_pretrained=args.load_pretrained,
                vocab_size=tok_vocab_size,
            )
            mdl = cast(TTTTransformerLM, mdl_raw)
            cfg = cast(GPT2Config, cfg_raw)
        else:
            mdl_raw, cfg_raw = load_ttt_model(
                model_name=model_name,
                fast_weight_type="ttt",
                seed=seed,
                load_pretrained=args.load_pretrained,
                vocab_size=tok_vocab_size,
            )
            # Cast to GPT-2 types (this script is GPT-2 only)
            mdl = cast(TTTTransformerLM, mdl_raw)
            cfg = cast(GPT2Config, cfg_raw)
        print(f"  Model config vocab_size: {cfg.vocab_size}")
        print(f"  Model embedding shape: {mdl.base_model.wte.embedding[...].shape}")
        return mdl, cfg

    # Set to training mode
    model, config = init_model(seeds[0])
    # ... (Optim creation code follows below in loop, but we need it for count_params above loop? 
    # No, count_params uses model. The original code had optimizer creation outside for printing.
    # We'll keep original flow but adapt resume)
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

    # Get action configuration
    action_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)
    
    # Scale learning rate inversely with number of updates to prevent overfitting
    # UPDATE_1: lr = 3e-4, UPDATE_2: lr = 1.5e-4, UPDATE_4: lr = 0.75e-4
    effective_lr = args.learning_rate / max(action_steps, 1)
    
    # Create optimizer for all parameters
    # Base model will be frozen via stop_gradient in the model's forward pass
    def create_optimizer():
        return nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(effective_lr),
            ),
            wrt=nnx.All(nnx.Param),
        )
    
    print(f"OK Optimizer: Adam (base_lr={args.learning_rate}, effective_lr={effective_lr}, scaled by 1/{max(action_steps, 1)})")
    print("   Base model frozen via stop_gradient")

    # Training loop
    print("\nStarting training...")
    print(f"Processing {args.max_chunks} chunks...")

    seed_results = []

    for seed in seeds:
        model, config = init_model(seed)

        # Set training mode only if we're actually doing updates
        # For SKIP action, use eval mode to disable dropout since we don't update
        if action_steps > 0:
            model.train()
        else:
            model.eval()
            print("  Using eval mode for SKIP action (no updates, disable dropout)")

        # Create optimizer once per seed
        optimizer = create_optimizer()
        
        start_chunk = 0
        
        # Resume Logic
        if args.resume_from and len(seeds) == 1:
            print(f"Resuming from checkpoint: {args.resume_from}")
            load_target = {"state": {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}}
            ckpt = load_checkpoint(args.resume_from, target=load_target)
            nnx.update(model, ckpt["state"]["model"])
            nnx.update(optimizer, ckpt["state"]["optimizer"])
            
            if "metadata" in ckpt and "chunks" in ckpt["metadata"]:
                start_chunk = ckpt["metadata"]["chunks"]
            elif "step" in ckpt:
                start_chunk = ckpt["step"]
                
            print(f"Resumed from chunk {start_chunk}")
            
        data_iter = create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            batch_size=batch_size,
            seq_length=seq_length,
            chunk_size=chunk_size,
            max_examples=examples_needed,
            num_workers=args.num_workers,
        )

        # Data skipping logic for resume
        chunks_per_seq = seq_length // chunk_size
        batches_to_skip = start_chunk // chunks_per_seq
        remainder_chunks = start_chunk % chunks_per_seq
        
        if batches_to_skip > 0:
            print(f"Skipping {batches_to_skip} batches to resume from chunk {start_chunk}...")
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    print("Warning: Data iterator exhausted during skipping!")
                    break
        
        first_batch = True

        total_loss_ce = 0.0
        total_loss_total = 0.0
        total_cost = 0.0
        chunks_processed = start_chunk

        print(f"\n=== Running seed {seed} ===")

        with tqdm(total=args.max_chunks, initial=start_chunk, desc=f"Training seed {seed}") as pbar:
            while chunks_processed < args.max_chunks:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    print("\nData iterator exhausted")
                    break

                num_chunks_available = batch["chunks"].shape[1]
                for chunk_idx in range(num_chunks_available):
                    # Skip already processed chunks in the first batch after resume
                    if first_batch and chunk_idx < remainder_chunks:
                        continue

                    if chunks_processed >= args.max_chunks:
                        break

                    chunk_batch = {
                        "input_ids": batch["chunks"][:, chunk_idx, :],
                        "attention_mask": batch["chunk_attention_mask"][:, chunk_idx, :],
                        "position_ids": jnp.arange(
                            chunk_idx * chunk_size, 
                            (chunk_idx + 1) * chunk_size, 
                            dtype=jnp.int32
                        )[None, :].repeat(batch["chunks"].shape[0], axis=0)
                    }

                    # Check for valid tokens
                    num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
                    if num_valid_tokens < 16:
                        continue

                    if action_steps == 0:
                        metrics = run_chunk_step(
                            model,
                            None,
                            chunk_batch,
                            use_ttt=False,
                            apply_update=False,
                            ssl_weight=0.0,
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
                                ssl_weight=args.ssl_weight,
                            )

                    assert metrics is not None
                    
                    # Check for stability
                    if not jnp.isfinite(metrics["loss_ce"]) or metrics["loss_ce"] > 20.0:
                        print(f"Warning: Skipping unstable chunk (loss={metrics['loss_ce']:.4f})")
                        continue
                        
                    total_loss_ce += metrics["loss_ce"]
                    total_loss_total += metrics["loss_total"]
                    total_cost += cost_multiplier
                    # Count sample-chunks processed (batch_size samples per position)
                    chunks_processed += batch_size

                    pbar.set_postfix(
                        {
                            "loss_ce": f"{metrics['loss_ce']:.4f}",
                            "loss_total": f"{metrics['loss_total']:.4f}",
                            "ppl": f"{metrics['perplexity']:.2f}",
                        }
                    )
                    pbar.update(batch_size)
                    
                    # WandB Log
                    if args.wandb_project:
                        wandb.log({
                            f"seed_{seed}/loss_total": metrics['loss_total'],
                            f"seed_{seed}/loss_ce": metrics['loss_ce'],
                            f"seed_{seed}/loss_aux": metrics['loss_aux'],
                            f"seed_{seed}/perplexity": metrics['perplexity'],
                            "chunks": chunks_processed,
                        })

                    if chunks_processed % 10 == 0:
                        denom = chunks_processed - start_chunk + 1e-6
                        avg_ce_loss = total_loss_ce / denom
                        avg_total_loss = total_loss_total / denom
                        avg_ppl = math.exp(avg_ce_loss)
                        print(f"\nChunk {chunks_processed}/{args.max_chunks}:")
                        print(f"  Average CE loss (since start): {avg_ce_loss:.4f}")
                        print(f"  Average total loss (since start): {avg_total_loss:.4f}")
                        print(f"  Average perplexity: {avg_ppl:.2f}")
                        
                    # Periodic Checkpoint
                    if chunks_processed % args.save_every == 0 and chunks_processed < args.max_chunks:
                        checkpoint_dir = output_dir / "checkpoints"
                        print(f"Saving checkpoint at chunk {chunks_processed}...")
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            step=chunks_processed,
                            state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                            metadata={"chunks": chunks_processed},
                        )
                
                first_batch = False

        if chunks_processed > 0:
            denom = chunks_processed - start_chunk + 1e-6
            final_avg_ce_loss = total_loss_ce / denom
            final_avg_total_loss = total_loss_total / denom
            final_avg_ppl = math.exp(final_avg_ce_loss)

            seed_results.append(
                {
                    'seed': seed,
                    'chunks_processed': chunks_processed,
                    'final_loss': final_avg_ce_loss,
                    'final_loss_ce': final_avg_ce_loss,
                    'final_loss_total': final_avg_total_loss,
                    'final_perplexity': final_avg_ppl,
                    'total_cost': total_cost,
                    'avg_cost_per_chunk': total_cost / chunks_processed,
                }
            )

            results = {
                'model_scale': args.model_scale,
                'action': args.action,
                'fast_weight_type': args.fast_weight_type,
                'lora_rank': args.lora_rank if args.fast_weight_type == 'lora' else None,
                'num_ttt_steps': num_ttt_steps,
                'cost_multiplier': cost_multiplier,
                'chunks_processed': chunks_processed,
                'final_loss': final_avg_ce_loss,
                'final_loss_ce': final_avg_ce_loss,
                'final_loss_total': final_avg_total_loss,
                'final_perplexity': final_avg_ppl,
                'learning_rate': args.learning_rate,
                'seed': seed,
                'total_cost': total_cost,
                'avg_cost_per_chunk': total_cost / chunks_processed,
            }

            suffix = f"_{args.fast_weight_type}"
            if args.fast_weight_type == "lora":
                suffix += f"_r{args.lora_rank}"
            results_file = output_dir / f"results_{args.model_scale}_{args.action}{suffix}_seed{seed}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nOK Results saved to: {results_file}")

            save_checkpoint(
                checkpoint_dir=output_dir / "checkpoints",
                step=chunks_processed,
                state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                metadata=results,
            )
            wait_for_checkpoints()
            print(f"OK Checkpoint saved to {output_dir / 'checkpoints'}")

    # Aggregate across seeds if multiple
    if seed_results:
        if len(seed_results) > 1:
            losses = jnp.array([r['final_loss'] for r in seed_results])
            perplexities = jnp.array([r['final_perplexity'] for r in seed_results])
            from ponderttt.utils.statistics import bootstrap_ci, compute_iqm
            loss_ci = bootstrap_ci(losses, n_bootstrap=1000)
            ppl_ci = bootstrap_ci(perplexities, n_bootstrap=1000)
            summary = {
                "seeds": [r['seed'] for r in seed_results],
                "loss_mean": float(losses.mean()),
                "loss_iqm": compute_iqm(losses),
                "loss_ci": loss_ci,
                "ppl_mean": float(perplexities.mean()),
                "ppl_iqm": compute_iqm(perplexities),
                "ppl_ci": ppl_ci,
            }
            summary_file = output_dir / f"summary_{args.model_scale}_{args.action}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSeed summary saved to {summary_file}")
        else:
            print("\nSingle-seed run complete.")
    else:
        print("\nNo chunks processed!")


if __name__ == "__main__":
    main()
