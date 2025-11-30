"""
Train Gemma 3 TTT models with multi-host TPU support.

PonderTTT Phase 2: Gemma 3 (4B, 12B) integration with Test-Time Training.

Supports:
- Single-host multi-device (e.g., 8 TPU cores)
- Multi-host multi-device (e.g., TPU v4-64 = 8 hosts × 8 cores)

Architecture:
- Slow weights (theta_slow): Frozen pretrained Gemma 3 backbone
- Fast weights (theta_fast): Adaptive TTT layer weights

Usage:
    # Single device / auto-sharding:
    python -m ponderttt.experiments.train_gemma3 \
        --model_scale 4b \
        --action UPDATE_1

    # Multi-host TPU with explicit sharding:
    python -m ponderttt.experiments.train_gemma3 \
        --model_scale 4b \
        --action UPDATE_1 \
        --enable_sharding \
        --ici_fsdp_parallelism 4 \
        --ici_tensor_parallelism 2

    # TPU v4-64 (8 hosts × 8 cores):
    python -m ponderttt.experiments.train_gemma3 \
        --model_scale 12b \
        --action UPDATE_1 \
        --enable_sharding \
        --dcn_data_parallelism 8 \
        --ici_fsdp_parallelism 4 \
        --ici_tensor_parallelism 2
"""

import argparse
import json
import logging
import math
from pathlib import Path
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model
from ..models.gemma3 import (
    Gemma3Config,
    ShardingConfig,
    create_device_mesh,
    get_data_sharding,
)
from ..utils.checkpointing import save_checkpoint, wait_for_checkpoints, load_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # type: ignore[invalid-assignment]
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gemma 3 TTT model with multi-host TPU support"
    )

    # Model configuration
    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["1b", "4b", "12b"],
        default="4b",
        help="Model scale: 1b (test), 4b, 12b",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint. Orbax: '/path/to/ckpt', HuggingFace: 'hf:google/gemma-3-4b-pt'",
    )

    # Training configuration
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
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--seq_length", type=int, default=4096, help="Sequence length"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512, help="Chunk size for TTT processing"
    )

    # Sharding configuration
    parser.add_argument(
        "--enable_sharding",
        action="store_true",
        help="Enable explicit multi-device sharding",
    )
    parser.add_argument(
        "--dcn_data_parallelism", type=int, default=-1,
        help="DCN (inter-host) data parallelism (-1 for auto)"
    )
    parser.add_argument(
        "--dcn_fsdp_parallelism", type=int, default=1,
        help="DCN FSDP parallelism"
    )
    parser.add_argument(
        "--dcn_tensor_parallelism", type=int, default=1,
        help="DCN tensor parallelism"
    )
    parser.add_argument(
        "--ici_data_parallelism", type=int, default=1,
        help="ICI (intra-host) data parallelism"
    )
    parser.add_argument(
        "--ici_fsdp_parallelism", type=int, default=-1,
        help="ICI FSDP parallelism (-1 for auto)"
    )
    parser.add_argument(
        "--ici_tensor_parallelism", type=int, default=1,
        help="ICI tensor parallelism"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gemma3_ttt",
        help="Output directory",
    )

    # Training options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        default=True,
        help="Load pretrained weights",
    )
    parser.add_argument(
        "--no-load-pretrained",
        action="store_false",
        dest="load_pretrained",
        help="Don't load pretrained weights",
    )
    parser.add_argument(
        "--ssl_weight",
        type=float,
        default=0.1,
        help="Weight for SSL auxiliary loss",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for multi-seed runs",
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (if None, WandB is disabled)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save checkpoint every N chunks",
    )

    # Data
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for data downloading",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="google/gemma-2-2b",
        help="Tokenizer to use (HuggingFace model name)",
    )

    # Resume
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    return parser.parse_args()


def action_to_steps(action: str) -> int:
    """Convert action name to number of TTT steps."""
    return {"SKIP": 0, "UPDATE_1": 1, "UPDATE_2": 2, "UPDATE_4": 4}[action]


def action_to_cost(action: str) -> float:
    """Convert action name to computational cost multiplier."""
    return {"SKIP": 1.0, "UPDATE_1": 3.0, "UPDATE_2": 5.0, "UPDATE_4": 9.0}[action]


def get_model_name(model_scale: str) -> str:
    """Convert model scale to internal model name."""
    return {"1b": "gemma3-1b", "4b": "gemma3-4b", "12b": "gemma3-12b"}[model_scale]


def count_params(model: nnx.Module) -> int:
    """Count total parameters in NNX model."""
    state = nnx.state(model)
    return sum(x.size for x in jax.tree.leaves(state) if hasattr(x, 'size'))


def create_sharding_config(args) -> ShardingConfig:
    """Create sharding configuration from args."""
    return ShardingConfig(
        dcn_data_parallelism=args.dcn_data_parallelism,
        dcn_fsdp_parallelism=args.dcn_fsdp_parallelism,
        dcn_tensor_parallelism=args.dcn_tensor_parallelism,
        ici_data_parallelism=args.ici_data_parallelism,
        ici_fsdp_parallelism=args.ici_fsdp_parallelism,
        ici_tensor_parallelism=args.ici_tensor_parallelism,
    )


def make_train_step(ssl_weight: float):
    """Create training step function."""

    def train_step(model, optimizer, batch, use_ttt: bool):
        def loss_fn(model):
            outputs = model(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                use_ttt=use_ttt,
            )
            logits = outputs["logits"]
            ttt_stats = outputs.get("ttt_stats", {})

            # Cross-entropy loss
            logits_for_loss = logits[:, :-1]
            labels = batch["input_ids"][:, 1:]
            mask = batch["attention_mask"][:, 1:] if "attention_mask" in batch else None

            # Compute CE loss
            log_probs = jax.nn.log_softmax(logits_for_loss, axis=-1)
            one_hot = jax.nn.one_hot(labels, logits_for_loss.shape[-1])
            ce_loss = -jnp.sum(log_probs * one_hot, axis=-1)

            if mask is not None:
                ce_loss = jnp.sum(ce_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            else:
                ce_loss = jnp.mean(ce_loss)

            # SSL auxiliary loss
            aux_loss = jnp.array(0.0)
            if use_ttt and ssl_weight > 0 and ttt_stats:
                ssl_terms = [
                    ttt_stats.get("ttt_loss_init"),
                    ttt_stats.get("ttt_loss_step_0"),
                    ttt_stats.get("ttt_loss_step_1"),
                ]
                ssl_values = [x.mean() for x in ssl_terms if x is not None]
                if ssl_values:
                    ssl_loss = sum(ssl_values) / len(ssl_values)
                    aux_loss = ssl_weight * ssl_loss

            total_loss = ce_loss + aux_loss
            return total_loss, (ce_loss, aux_loss, ttt_stats)

        (loss, (ce_loss, aux_loss, ttt_stats)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        optimizer.update(model, grads)

        perplexity = jnp.exp(ce_loss)
        metrics = {
            "loss_total": loss,
            "loss_ce": ce_loss,
            "loss_aux": aux_loss,
            "perplexity": perplexity,
        }
        return metrics, ttt_stats

    return train_step


def make_eval_step():
    """Create evaluation step function."""

    def eval_step(model, batch, use_ttt: bool):
        outputs = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            use_ttt=use_ttt,
        )
        logits = outputs["logits"]

        logits_for_loss = logits[:, :-1]
        labels = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:] if "attention_mask" in batch else None

        log_probs = jax.nn.log_softmax(logits_for_loss, axis=-1)
        one_hot = jax.nn.one_hot(labels, logits_for_loss.shape[-1])
        ce_loss = -jnp.sum(log_probs * one_hot, axis=-1)

        if mask is not None:
            ce_loss = jnp.sum(ce_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            ce_loss = jnp.mean(ce_loss)

        return {"loss_ce": ce_loss, "perplexity": jnp.exp(ce_loss)}

    return eval_step


def main():
    args = parse_args()
    seeds = [args.seed] if args.seeds is None else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    logger.info("=" * 60)
    logger.info("PonderTTT Gemma 3 Training (Multi-Host TPU Support)")
    logger.info("=" * 60)
    logger.info(f"Model scale: {args.model_scale}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Max chunks: {args.max_chunks}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Enable sharding: {args.enable_sharding}")

    # JAX device info
    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} ({devices[0].platform})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup mesh for multi-device training
    mesh = None
    data_sharding = None

    if args.enable_sharding:
        sharding_config = create_sharding_config(args)
        mesh = create_device_mesh(sharding_config)
        data_sharding = get_data_sharding(mesh, sharding_config)
        logger.info(f"Mesh created: {mesh.shape}")
        logger.info(f"Data sharding: {data_sharding}")

    # Get model configuration
    model_name = get_model_name(args.model_scale)
    num_ttt_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)
    use_ttt = num_ttt_steps > 0

    logger.info("\nConfiguration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  TTT steps: {num_ttt_steps}")
    logger.info(f"  Cost multiplier: {cost_multiplier}x")
    logger.info(f"  Use TTT: {use_ttt}")
    if args.checkpoint_path:
        logger.info(f"  Checkpoint: {args.checkpoint_path}")

    # Initialize WandB
    if args.wandb_project and WANDB_AVAILABLE and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"gemma3_{args.model_scale}_{args.action}",
        )

    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {args.tokenizer_name}...")
    tokenizer = get_tokenizer(args.tokenizer_name)

    # Calculate data requirements
    batch_size = args.batch_size
    seq_length = args.seq_length
    chunk_size = args.chunk_size
    chunks_per_sequence = seq_length // chunk_size
    examples_needed = math.ceil(args.max_chunks / max(chunks_per_sequence, 1))

    # Create train/eval step functions
    train_step_fn = make_train_step(args.ssl_weight)
    eval_step_fn = make_eval_step()

    def init_model(seed):
        logger.info(f"\nInitializing Gemma 3 model with TTT layer (seed={seed})...")
        model, config = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            dtype=jnp.bfloat16,
            seed=seed,
            load_pretrained=args.load_pretrained,
            checkpoint_path=args.checkpoint_path,
        )
        return model, config

    # Initialize first model for printing stats
    model, config = init_model(seeds[0])
    gemma_config = cast(Gemma3Config, config)

    logger.info("\nModel loaded:")
    logger.info(f"  Layers: {gemma_config.num_layers}")
    logger.info(f"  Hidden dim: {gemma_config.embed_dim}")
    logger.info(f"  Heads: {gemma_config.num_heads} (KV: {gemma_config.num_kv_heads})")

    total_params = count_params(model)
    trainable_state = model.get_trainable_params()
    trainable_params = sum(x.size for x in jax.tree.leaves(trainable_state) if hasattr(x, 'size'))

    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters (TTT layer): {trainable_params:,}")
    logger.info(f"  Frozen parameters (backbone): {total_params - trainable_params:,}")

    # Optimizer setup
    action_steps = action_to_steps(args.action)
    effective_lr = args.learning_rate / max(action_steps, 1)

    def create_optimizer(mdl):
        return nnx.Optimizer(
            mdl,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(effective_lr),
            ),
            wrt=nnx.All(nnx.Param),
        )

    logger.info(f"\nOptimizer: Adam (lr={effective_lr:.2e})")
    logger.info("  Gemma 3 backbone frozen via stop_gradient")

    # JIT compile train/eval steps
    if mesh is not None:
        # With explicit sharding
        jit_train_step = partial(jax.jit, static_argnames=['use_ttt'])(
            lambda model, optimizer, batch, use_ttt: train_step_fn(model, optimizer, batch, use_ttt)
        )
        jit_eval_step = partial(jax.jit, static_argnames=['use_ttt'])(
            lambda model, batch, use_ttt: eval_step_fn(model, batch, use_ttt)
        )
    else:
        # Auto sharding
        jit_train_step = nnx.jit(
            lambda model, optimizer, batch, use_ttt: train_step_fn(model, optimizer, batch, use_ttt),
            static_argnames=['use_ttt']
        )
        jit_eval_step = nnx.jit(
            lambda model, batch, use_ttt: eval_step_fn(model, batch, use_ttt),
            static_argnames=['use_ttt']
        )

    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Processing {args.max_chunks} chunks...")

    seed_results = []

    for seed in seeds:
        model, config = init_model(seed)

        if action_steps > 0:
            model.train()
        else:
            model.eval()
            logger.info("  Using eval mode for SKIP action")

        optimizer = create_optimizer(model)
        start_chunk = 0

        # Resume logic
        if args.resume_from and len(seeds) == 1:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            load_target = {"state": {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}}
            ckpt = load_checkpoint(args.resume_from, target=load_target)
            nnx.update(model, ckpt["state"]["model"])
            nnx.update(optimizer, ckpt["state"]["optimizer"])

            if "metadata" in ckpt and "chunks" in ckpt["metadata"]:
                start_chunk = ckpt["metadata"]["chunks"]
            elif "step" in ckpt:
                start_chunk = ckpt["step"]

            logger.info(f"Resumed from chunk {start_chunk}")

        # Create data iterator
        data_iter = create_data_iterator(
            tokenizer=tokenizer,
            split="train",
            batch_size=batch_size,
            seq_length=seq_length,
            chunk_size=chunk_size,
            max_examples=examples_needed * batch_size,
            num_workers=args.num_workers,
        )

        # Skip data for resume
        chunks_per_seq = seq_length // chunk_size
        batches_to_skip = start_chunk // chunks_per_seq
        remainder_chunks = start_chunk % chunks_per_seq

        if batches_to_skip > 0:
            logger.info(f"Skipping {batches_to_skip} batches...")
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    break

        first_batch = True
        total_loss_ce = 0.0
        total_loss_total = 0.0
        total_cost = 0.0
        chunks_processed = start_chunk

        logger.info(f"\n=== Running seed {seed} ===")

        with tqdm(total=args.max_chunks, initial=start_chunk, desc=f"Training seed {seed}") as pbar:
            while chunks_processed < args.max_chunks:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    logger.info("\nData iterator exhausted")
                    break

                num_chunks_available = batch["chunks"].shape[1]
                metrics: dict[str, jnp.ndarray] = {}
                for chunk_idx in range(num_chunks_available):
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

                    # Apply data sharding if enabled
                    if data_sharding is not None:
                        chunk_batch = jax.tree.map(
                            lambda x: jax.device_put(x, data_sharding),
                            chunk_batch
                        )

                    # Check for valid tokens
                    num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
                    if num_valid_tokens < 16:
                        continue

                    if action_steps == 0:
                        metrics = jit_eval_step(model, chunk_batch, use_ttt=False)
                        metrics["loss_total"] = metrics["loss_ce"]
                        metrics["loss_aux"] = jnp.array(0.0)
                    else:
                        for _ in range(action_steps):
                            metrics, _ = jit_train_step(
                                model, optimizer, chunk_batch, use_ttt=True
                            )

                    # Stability check
                    if not metrics:
                        continue
                    loss_ce = float(metrics["loss_ce"])
                    if not jnp.isfinite(loss_ce) or loss_ce > 20.0:
                        logger.warning(f"Skipping unstable chunk (loss={loss_ce:.4f})")
                        continue

                    total_loss_ce += loss_ce
                    total_loss_total += float(metrics["loss_total"])
                    total_cost += cost_multiplier
                    chunks_processed += 1

                    pbar.set_postfix({
                        "loss_ce": f"{loss_ce:.4f}",
                        "ppl": f"{float(metrics['perplexity']):.2f}",
                    })
                    pbar.update(1)

                    # WandB logging
                    if args.wandb_project and WANDB_AVAILABLE and wandb is not None:
                        wandb.log({
                            f"seed_{seed}/loss_total": float(metrics['loss_total']),
                            f"seed_{seed}/loss_ce": loss_ce,
                            f"seed_{seed}/loss_aux": float(metrics['loss_aux']),
                            f"seed_{seed}/perplexity": float(metrics['perplexity']),
                            "chunks": chunks_processed,
                        })

                    if chunks_processed % 10 == 0:
                        denom = chunks_processed - start_chunk + 1e-6
                        avg_ce_loss = total_loss_ce / denom
                        avg_ppl = math.exp(avg_ce_loss)
                        logger.info(f"\nChunk {chunks_processed}/{args.max_chunks}:")
                        logger.info(f"  Avg CE loss: {avg_ce_loss:.4f}, Avg PPL: {avg_ppl:.2f}")

                    # Periodic checkpoint
                    if chunks_processed % args.save_every == 0 and chunks_processed < args.max_chunks:
                        checkpoint_dir = output_dir / "checkpoints"
                        logger.info(f"Saving checkpoint at chunk {chunks_processed}...")
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            step=chunks_processed,
                            state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                            metadata={"chunks": chunks_processed},
                        )

                first_batch = False

        # Save results
        if chunks_processed > 0:
            denom = chunks_processed - start_chunk + 1e-6
            final_avg_ce_loss = total_loss_ce / denom
            final_avg_ppl = math.exp(final_avg_ce_loss)

            seed_results.append({
                'seed': seed,
                'chunks_processed': chunks_processed,
                'final_loss': final_avg_ce_loss,
                'final_perplexity': final_avg_ppl,
                'total_cost': total_cost,
            })

            results = {
                'model_scale': args.model_scale,
                'model_name': model_name,
                'action': args.action,
                'num_ttt_steps': num_ttt_steps,
                'cost_multiplier': cost_multiplier,
                'chunks_processed': chunks_processed,
                'final_loss': final_avg_ce_loss,
                'final_perplexity': final_avg_ppl,
                'learning_rate': args.learning_rate,
                'seed': seed,
                'total_cost': total_cost,
                'seq_length': args.seq_length,
                'chunk_size': args.chunk_size,
                'enable_sharding': args.enable_sharding,
                'num_devices': len(jax.devices()),
            }

            results_file = output_dir / f"results_gemma3_{args.model_scale}_{args.action}_seed{seed}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to: {results_file}")

            # Save final checkpoint
            save_checkpoint(
                checkpoint_dir=output_dir / "checkpoints",
                step=chunks_processed,
                state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                metadata=results,
            )
            wait_for_checkpoints()
            logger.info(f"Checkpoint saved to {output_dir / 'checkpoints'}")

    # Aggregate multi-seed results
    if seed_results and len(seed_results) > 1:
        losses = jnp.array([r['final_loss'] for r in seed_results])
        perplexities = jnp.array([r['final_perplexity'] for r in seed_results])
        from ..utils.statistics import bootstrap_ci, compute_iqm

        summary = {
            "seeds": [r['seed'] for r in seed_results],
            "loss_mean": float(losses.mean()),
            "loss_iqm": compute_iqm(losses),
            "loss_ci": bootstrap_ci(losses, n_bootstrap=1000),
            "ppl_mean": float(perplexities.mean()),
            "ppl_iqm": compute_iqm(perplexities),
            "ppl_ci": bootstrap_ci(perplexities, n_bootstrap=1000),
        }
        summary_file = output_dir / f"summary_gemma3_{args.model_scale}_{args.action}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSeed summary saved to {summary_file}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
