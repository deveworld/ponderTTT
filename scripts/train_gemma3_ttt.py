#!/usr/bin/env python3
"""
Train Gemma 3 TTT models with frozen backbone.

This script is specifically designed for Gemma 3 models (1B, 4B, 12B, 27B)
with frozen base model training - only TTT layer parameters are updated.

Usage:
    python scripts/train_gemma3_ttt.py --model_scale 4b --action UPDATE_1
    python scripts/train_gemma3_ttt.py --model_scale 12b --action UPDATE_2 --enable_sharding

Requirements:
    - NVIDIA GPU with sufficient VRAM or TPU
    - Gemma 3 checkpoint (HuggingFace or Orbax format)
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from ponderttt.data import create_data_iterator, get_tokenizer
from ponderttt.models import load_ttt_model
from ponderttt.models.base_model_nnx import TTTModel
from ponderttt.utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    wait_for_checkpoints,
)

# Import Gemma 3 specific modules
try:
    from ponderttt.models.gemma3 import (
        ShardingConfig,
        create_device_mesh,
        get_data_sharding,
    )

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

# Optional WandB
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gemma 3 TTT models with frozen backbone"
    )

    # Model configuration
    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["1b", "4b", "12b", "27b"],
        default="4b",
        help="Gemma 3 model scale",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (HF format: 'hf:google/gemma-3-4b-pt')",
    )

    # Training action
    parser.add_argument(
        "--action",
        type=str,
        choices=["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4", "ADAPTIVE"],
        default="UPDATE_1",
        help="Training action (number of TTT steps per chunk)",
    )
    parser.add_argument(
        "--gating_threshold",
        type=float,
        default=2.0,
        help="Loss threshold for ADAPTIVE gating",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=160000,
        help="Maximum number of chunks to process",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length (Gemma 3 supports up to 128K)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for TTT",
    )
    parser.add_argument(
        "--ssl_weight",
        type=float,
        default=0.1,
        help="SSL auxiliary loss weight",
    )

    # Sharding configuration (for multi-host TPU)
    parser.add_argument(
        "--enable_sharding",
        action="store_true",
        help="Enable explicit multi-device FSDP sharding",
    )
    parser.add_argument(
        "--dcn_data_parallelism",
        type=int,
        default=-1,
        help="DCN (inter-host) data parallelism (-1 for auto)",
    )
    parser.add_argument(
        "--dcn_fsdp_parallelism",
        type=int,
        default=1,
        help="DCN FSDP parallelism",
    )
    parser.add_argument(
        "--ici_data_parallelism",
        type=int,
        default=1,
        help="ICI (intra-host) data parallelism",
    )
    parser.add_argument(
        "--ici_fsdp_parallelism",
        type=int,
        default=-1,
        help="ICI FSDP parallelism (-1 for auto)",
    )

    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gemma3",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10000,
        help="Save checkpoint every N chunks",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )

    # Data configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Data loading workers",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name (auto-detected if not specified)",
    )

    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def action_to_steps(action: str) -> int:
    """Convert action name to number of TTT steps."""
    if action == "ADAPTIVE":
        return 1
    return {"SKIP": 0, "UPDATE_1": 1, "UPDATE_2": 2, "UPDATE_4": 4}[action]


def action_to_cost(action: str) -> float:
    """Convert action name to computational cost multiplier."""
    if action == "ADAPTIVE":
        return 2.0
    return {"SKIP": 1.0, "UPDATE_1": 3.0, "UPDATE_2": 5.0, "UPDATE_4": 9.0}[action]


def get_model_name(model_scale: str) -> str:
    """Convert model scale to model name."""
    return f"gemma3-{model_scale}"


def get_default_tokenizer(model_scale: str) -> str:
    """Get default tokenizer for model scale."""
    return f"google/gemma-3-{model_scale}-pt"


def count_params(model: nnx.Module) -> int:
    """Count total parameters in NNX model."""
    state = nnx.state(model)
    return sum(x.size for x in jax.tree.leaves(state) if hasattr(x, "size"))


def get_base_model_checksum(model: TTTModel) -> float:
    """Calculate checksum of base model weights."""
    base = getattr(model, "base_model", None)
    if base is None:
        return 0.0
    base_params = nnx.state(base, nnx.Param)
    leaves = jax.tree.leaves(base_params)
    if not leaves:
        return 0.0
    return float(sum(float(jnp.sum(x)) for x in leaves))


# =============================================================================
# Trainable Parameters Wrapper
# =============================================================================


class TrainableParamsWrapper(nnx.Module):
    """Wrapper containing only trainable submodules for optimizer targeting.

    In NNX, nnx.Optimizer applies updates to all nnx.Param in the target module.
    By wrapping only the trainable submodules (TTT layer, fast_norm, projections),
    we ensure the optimizer only updates those parameters while the base model
    remains frozen.

    IMPORTANT: This wrapper shares the actual submodule objects with the model,
    so updates to the wrapper's parameters update the model's parameters.
    """

    def __init__(self, model: TTTModel):
        """Extract trainable submodules from model.

        Args:
            model: TTT model (Gemma3TTTModel or similar)
        """
        # These are references to the actual submodules, not copies
        self.fast_layer = model.fast_layer
        self.fast_norm = model.fast_norm

        # Handle projection layers if present (Gemma3TTTModel)
        self.has_projections = getattr(model, "_needs_projection", False)
        if self.has_projections:
            self.ttt_proj_in = model.ttt_proj_in
            self.ttt_proj_out = model.ttt_proj_out


# =============================================================================
# Training Step Functions
# =============================================================================


def make_train_step(ssl_weight: float) -> Callable:
    """Create training step function for Gemma 3.

    The gradient is computed w.r.t. trainable_wrapper, but the forward pass
    uses the full model. Since trainable_wrapper shares submodules with model,
    the gradients flow correctly through the trainable parameters.
    """

    def train_step(
        model: TTTModel,
        trainable_wrapper: TrainableParamsWrapper,
        optimizer: nnx.Optimizer,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_ttt: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        def loss_fn(
            wrapper: TrainableParamsWrapper,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, dict]]:
            # Forward pass through full model (wrapper shares submodules with model)
            # Gemma3TTTModel returns (outputs_dict, cache)
            outputs, _ = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_ttt=use_ttt,
            )
            logits = outputs["logits"]
            ttt_stats = outputs.get("ttt_stats", {})

            # Cross-entropy loss
            logits_for_loss = logits[:, :-1]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, 1:]

            log_probs = jax.nn.log_softmax(logits_for_loss, axis=-1)
            one_hot = jax.nn.one_hot(labels, logits_for_loss.shape[-1])
            ce_loss = -jnp.sum(log_probs * one_hot, axis=-1)
            ce_loss = jnp.sum(ce_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            # SSL auxiliary loss
            aux_loss = jnp.array(0.0)
            if use_ttt and ssl_weight > 0 and ttt_stats:
                ssl_terms = [
                    ttt_stats.get("ttt_loss_init"),
                    ttt_stats.get("ttt_loss_step_0"),
                    ttt_stats.get("ttt_loss_step_1"),
                ]
                ssl_values = [jnp.mean(x) for x in ssl_terms if x is not None]
                if ssl_values:
                    ssl_loss = sum(ssl_values) / len(ssl_values)
                    aux_loss = jnp.asarray(ssl_weight * ssl_loss)

            total_loss = ce_loss + aux_loss
            return total_loss, (ce_loss, aux_loss, ttt_stats)

        # Compute gradient w.r.t. trainable wrapper only
        (loss, (ce_loss, aux_loss, ttt_stats)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(trainable_wrapper)

        # Update only trainable params via optimizer (targets wrapper)
        optimizer.update(trainable_wrapper, grads)

        metrics = {
            "loss_total": loss,
            "loss_ce": ce_loss,
            "loss_aux": aux_loss,
            "perplexity": jnp.exp(ce_loss),
        }
        return metrics, ttt_stats

    return train_step


def make_eval_step() -> Callable:
    """Create evaluation step function (SKIP action) for Gemma 3."""

    def eval_step(
        model: TTTModel,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_ttt: bool,
    ) -> dict[str, jax.Array]:
        # Gemma3TTTModel returns (outputs_dict, cache)
        outputs, _ = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_ttt=use_ttt,
        )
        logits = outputs["logits"]

        logits_for_loss = logits[:, :-1]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        log_probs = jax.nn.log_softmax(logits_for_loss, axis=-1)
        one_hot = jax.nn.one_hot(labels, logits_for_loss.shape[-1])
        ce_loss = -jnp.sum(log_probs * one_hot, axis=-1)
        ce_loss = jnp.sum(ce_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        return {"loss_ce": ce_loss, "perplexity": jnp.exp(ce_loss)}

    return eval_step


# =============================================================================
# Main Training Function
# =============================================================================


def main() -> None:
    args = parse_args()

    if not GEMMA3_AVAILABLE:
        raise RuntimeError("Gemma 3 modules not available. Install gemma dependencies.")

    # Initialize distributed for multi-host
    if args.enable_sharding:
        jax.distributed.initialize()

    # Logging
    logger.info("=" * 60)
    logger.info("PonderTTT Gemma 3 Training")
    logger.info("=" * 60)
    logger.info(f"Model scale: Gemma 3 {args.model_scale.upper()}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Max chunks: {args.max_chunks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Chunk size: {args.chunk_size}")

    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} ({devices[0].platform})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup mesh for multi-device
    mesh = None
    data_sharding = None
    if args.enable_sharding:
        sharding_config = ShardingConfig(
            dcn_data_parallelism=args.dcn_data_parallelism,
            dcn_fsdp_parallelism=args.dcn_fsdp_parallelism,
            ici_data_parallelism=args.ici_data_parallelism,
            ici_fsdp_parallelism=args.ici_fsdp_parallelism,
        )
        mesh = create_device_mesh(sharding_config)
        data_sharding = get_data_sharding(mesh, sharding_config)
        logger.info(f"Mesh: {mesh.shape}")
        logger.info(f"Data sharding: {data_sharding}")

    # Model configuration
    model_name = get_model_name(args.model_scale)
    num_ttt_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)

    logger.info(f"\nModel: {model_name}")
    logger.info(f"TTT steps: {num_ttt_steps}, Cost: {cost_multiplier}x")

    # Initialize WandB
    if args.wandb_project and WANDB_AVAILABLE and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"train_gemma3_{args.model_scale}_{args.action}",
        )

    # Load tokenizer
    tokenizer_name = args.tokenizer_name or get_default_tokenizer(args.model_scale)
    logger.info(f"\nTokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)

    # Calculate data requirements
    seq_length = args.seq_length
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    chunks_per_seq = seq_length // chunk_size
    examples_needed = math.ceil(args.max_chunks / max(chunks_per_seq, 1))

    # Create training step functions
    train_step_fn = make_train_step(args.ssl_weight)
    eval_step_fn = make_eval_step()

    # Load model
    logger.info(f"\nLoading model from: {args.checkpoint_path or 'random init'}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        # Default HuggingFace path
        checkpoint_path = f"hf:google/gemma-3-{args.model_scale}-pt"

    model, config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        dtype=jnp.bfloat16,
        seed=args.seed,
        load_pretrained=True,
        checkpoint_path=checkpoint_path,
    )

    # Count parameters
    total_params = count_params(model)
    trainable_state = model.get_trainable_params()
    trainable_params = sum(
        x.size for x in jax.tree.leaves(trainable_state) if hasattr(x, "size")
    )

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable (TTT): {trainable_params:,}")
    logger.info(f"Frozen: {total_params - trainable_params:,}")

    # Verify frozen base model
    initial_checksum = get_base_model_checksum(model)
    logger.info(f"Base model checksum: {initial_checksum:.4f}")

    # Create trainable wrapper for optimizer (freezes base model)
    trainable_wrapper = TrainableParamsWrapper(model)

    # Optimizer (targets only trainable wrapper)
    action_steps = action_to_steps(args.action)
    effective_lr = args.learning_rate / max(action_steps, 1)

    optimizer = nnx.Optimizer(
        trainable_wrapper,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(effective_lr),
        ),
    )

    logger.info(f"\nOptimizer: Adam (lr={effective_lr:.2e})")
    logger.info("Optimizer targets: TTT layer + fast_norm + projections only")

    # JIT compile step functions
    jit_train_step = nnx.jit(train_step_fn, static_argnames=["use_ttt"])
    jit_eval_step = nnx.jit(eval_step_fn, static_argnames=["use_ttt"])

    # Set training mode
    if action_steps > 0:
        model.train()
    else:
        model.eval()

    start_chunk = 0

    # Resume logic
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from}")
        ckpt = load_checkpoint(
            args.resume_from,
            target={
                "state": {
                    "model": nnx.state(model),
                    "optimizer": nnx.state(optimizer),
                }
            },
        )
        nnx.update(model, ckpt["state"]["model"])
        nnx.update(optimizer, ckpt["state"]["optimizer"])
        start_chunk = ckpt.get("metadata", {}).get("chunks", ckpt.get("step", 0))
        logger.info(f"Resumed from chunk {start_chunk}")

    # Create data iterator
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=batch_size,
        seq_length=seq_length,
        chunk_size=chunk_size,
        max_examples=examples_needed,
        num_workers=args.num_workers,
    )

    # Skip for resume
    chunks_per_batch = batch_size * chunks_per_seq
    batches_to_skip = start_chunk // chunks_per_batch

    for _ in range(batches_to_skip):
        try:
            next(data_iter)
        except StopIteration:
            break

    # Training loop
    logger.info("\nStarting training...")

    total_loss_ce = 0.0
    total_loss_total = 0.0
    total_cost = 0.0
    chunks_processed = start_chunk

    with tqdm(total=args.max_chunks, initial=start_chunk, desc="Training") as pbar:
        while chunks_processed < args.max_chunks:
            try:
                batch = next(data_iter)
            except StopIteration:
                logger.info("Data exhausted")
                break

            num_chunks_in_batch = batch["chunks"].shape[1]

            for chunk_idx in range(num_chunks_in_batch):
                if chunks_processed >= args.max_chunks:
                    break

                chunk_input_ids = batch["chunks"][:, chunk_idx, :]
                chunk_mask = batch["chunk_attention_mask"][:, chunk_idx, :]
                chunk_pos = jnp.arange(chunk_size, dtype=jnp.int32)[None, :].repeat(
                    batch_size, axis=0
                )

                # Apply sharding
                if data_sharding is not None:
                    chunk_input_ids = jax.device_put(chunk_input_ids, data_sharding)
                    chunk_mask = jax.device_put(chunk_mask, data_sharding)
                    chunk_pos = jax.device_put(chunk_pos, data_sharding)

                # Skip if too few valid tokens
                if jnp.sum(chunk_mask[:, 1:]) < 16:
                    continue

                # Execute step
                if action_steps == 0:
                    metrics = jit_eval_step(
                        model, chunk_input_ids, chunk_mask, chunk_pos, use_ttt=False
                    )
                    metrics["loss_total"] = metrics["loss_ce"]
                    metrics["loss_aux"] = jnp.array(0.0)
                else:
                    for _ in range(action_steps):
                        metrics, _ = jit_train_step(
                            model,
                            trainable_wrapper,
                            optimizer,
                            chunk_input_ids,
                            chunk_mask,
                            chunk_pos,
                            use_ttt=True,
                        )

                # Check stability
                if not jnp.isfinite(metrics["loss_ce"]) or metrics["loss_ce"] > 20.0:
                    logger.warning(f"Unstable loss: {metrics['loss_ce']:.4f}")
                    continue

                total_loss_ce += float(metrics["loss_ce"])
                total_loss_total += float(metrics["loss_total"])
                total_cost += cost_multiplier
                chunks_processed += batch_size

                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss_ce']:.4f}",
                        "ppl": f"{jnp.exp(metrics['loss_ce']):.2f}",
                    }
                )
                pbar.update(batch_size)

                # WandB logging
                if args.wandb_project and WANDB_AVAILABLE and wandb is not None:
                    wandb.log(
                        {
                            "loss_ce": float(metrics["loss_ce"]),
                            "loss_total": float(metrics["loss_total"]),
                            "chunks": chunks_processed,
                        }
                    )

                # Periodic checkpoint
                if (
                    chunks_processed % args.save_every == 0
                    and chunks_processed < args.max_chunks
                ):
                    save_checkpoint(
                        checkpoint_dir=output_dir / "checkpoints",
                        step=chunks_processed,
                        state={
                            "model": nnx.state(model),
                            "optimizer": nnx.state(optimizer),
                        },
                        metadata={"chunks": chunks_processed},
                    )

    # Training complete
    if chunks_processed > 0:
        denom = chunks_processed - start_chunk + 1e-6
        final_loss = total_loss_ce / denom
        final_ppl = math.exp(final_loss)

        # Verify frozen weights
        final_checksum = get_base_model_checksum(model)
        if abs(final_checksum - initial_checksum) > 1e-4:
            logger.warning("⚠️ Base model weights changed!")
        else:
            logger.info("✓ Base model frozen successfully")

        # Save results
        results = {
            "model_scale": args.model_scale,
            "action": args.action,
            "chunks": chunks_processed,
            "loss": final_loss,
            "perplexity": final_ppl,
            "seed": args.seed,
        }
        results_file = (
            output_dir / f"results_gemma3_{args.model_scale}_{args.action}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")

        # Final checkpoint
        save_checkpoint(
            checkpoint_dir=output_dir / "checkpoints",
            step=chunks_processed,
            state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
            metadata=results,
        )
        wait_for_checkpoints()

        logger.info(f"\n{'=' * 60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Final Loss: {final_loss:.4f}")
        logger.info(f"Final Perplexity: {final_ppl:.2f}")
        logger.info(f"Chunks Processed: {chunks_processed:,}")
        logger.info(f"Total Cost: {total_cost:.2f}x")


if __name__ == "__main__":
    main()
