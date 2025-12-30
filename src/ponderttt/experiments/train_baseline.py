"""
Train TTT models with fixed action schedules.

Supports:
- GPT-2 models (gpt2, medium, large, xl)
- Gemma 3 models (1b, 4b, 12b, 27b)
- Single-host and multi-host TPU with explicit sharding

Architecture:
- Slow weights (theta_slow): Frozen pretrained model
- Fast weights (theta_fast): Adaptive TTT layer weights

Usage:
    # GPT-2 (single device):
    python -m ponderttt.experiments.train_baseline --model_scale gpt2 --action UPDATE_1

    # Gemma 3 (single device):
    python -m ponderttt.experiments.train_baseline --model_scale 4b --action UPDATE_1

    # Gemma 3 (multi-host TPU):
    python -m ponderttt.experiments.train_baseline \
        --model_scale 4b --action UPDATE_1 \
        --enable_sharding --ici_fsdp_parallelism 4
"""

import argparse
import json
import logging
import math
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTModel
from ..utils.checkpointing import save_checkpoint, wait_for_checkpoints, load_checkpoint

# Optional imports for Gemma 3 sharding
try:
    from ..models.gemma3 import (  # noqa: F401
        Gemma3Config,
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
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TTT model (GPT-2 or Gemma 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["gpt2", "medium", "large", "xl", "1b", "4b", "12b", "27b"],
        default="gpt2",
        help="Model scale: GPT-2 (gpt2/medium/large/xl) or Gemma 3 (1b/4b/12b/27b)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint (Orbax dir or 'hf:org/model' for HuggingFace)",
    )

    # Training configuration
    parser.add_argument(
        "--action",
        type=str,
        choices=["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4", "ADAPTIVE"],
        required=True,
        help="Action: SKIP, UPDATE_N, or ADAPTIVE (Loss Skip gating)",
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
        default=100,
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
        default=1024,
        help="Sequence length",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for TTT",
    )

    # Fast weight configuration
    parser.add_argument(
        "--fast_weight_type",
        type=str,
        choices=["ttt", "lora"],
        default="ttt",
        help="Fast weight type (LoRA only for GPT-2)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank (if fast_weight_type='lora')",
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
        help="Enable explicit multi-device sharding",
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
        "--dcn_tensor_parallelism",
        type=int,
        default=1,
        help="DCN tensor parallelism",
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
    parser.add_argument(
        "--ici_tensor_parallelism",
        type=int,
        default=1,
        help="ICI tensor parallelism",
    )

    # Output and logging
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
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed runs",
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
        default=1000,
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
        default=32,
        help="Data loading workers",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name (auto-detected if not specified)",
    )

    # Pretrained weights
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

    return parser.parse_args()


# =============================================================================
# Helper Functions
# =============================================================================


def is_gemma_model(model_scale: str) -> bool:
    """Check if model scale is Gemma 3."""
    return model_scale in ["1b", "4b", "12b", "27b"]


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
    mapping = {
        # GPT-2
        "gpt2": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl",
        # Gemma 3
        "1b": "gemma3-1b",
        "4b": "gemma3-4b",
        "12b": "gemma3-12b",
        "27b": "gemma3-27b",
    }
    return mapping[model_scale]


def get_default_tokenizer(model_scale: str) -> str:
    """Get default tokenizer for model scale."""
    if is_gemma_model(model_scale):
        return "google/gemma-3-1b-it"
    return get_model_name(model_scale)


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
# Training Step Factories
# =============================================================================


def make_train_step(ssl_weight: float) -> Callable:
    """Create training step function."""

    def train_step(
        model: TTTModel,
        optimizer: nnx.Optimizer,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_ttt: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        def loss_fn(
            mdl: TTTModel,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, dict]]:
            outputs = mdl(
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

        (loss, (ce_loss, aux_loss, ttt_stats)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        optimizer.update(model, grads)

        metrics = {
            "loss_total": loss,
            "loss_ce": ce_loss,
            "loss_aux": aux_loss,
            "perplexity": jnp.exp(ce_loss),
        }
        return metrics, ttt_stats

    return train_step


def make_eval_step() -> Callable:
    """Create evaluation step function (SKIP action)."""

    def eval_step(
        model: TTTModel,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_ttt: bool,
    ) -> dict[str, jax.Array]:
        outputs = model(
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


def make_adaptive_train_step(ssl_weight: float, threshold: float) -> Callable:
    """Create adaptive training step with loss-based gating."""
    base_train_step = make_train_step(ssl_weight)
    base_eval_step = make_eval_step()

    def adaptive_step(
        model: TTTModel,
        optimizer: nnx.Optimizer,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        use_ttt: bool,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        # Compute gating signal (TTT reconstruction loss)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_ttt=True,
        )
        ttt_stats = outputs.get("ttt_stats", {})

        gating_signal = 0.0
        if ttt_stats and "ttt_loss_step_0" in ttt_stats:
            gating_signal = float(jnp.mean(ttt_stats["ttt_loss_step_0"]))
        elif ttt_stats and "ttt_loss_init" in ttt_stats:
            gating_signal = float(jnp.mean(ttt_stats["ttt_loss_init"]))

        should_update = gating_signal > threshold

        if should_update:
            metrics, ttt_stats = base_train_step(
                model, optimizer, input_ids, attention_mask, position_ids, True
            )
            metrics["updated"] = jnp.array(1.0)
        else:
            metrics = base_eval_step(
                model, input_ids, attention_mask, position_ids, False
            )
            metrics["loss_total"] = metrics["loss_ce"]
            metrics["loss_aux"] = jnp.array(0.0)
            metrics["updated"] = jnp.array(0.0)
            ttt_stats = {}

        metrics["gating_signal"] = jnp.array(gating_signal)
        return metrics, ttt_stats

    return adaptive_step


# =============================================================================
# Main Training Function
# =============================================================================


def main() -> None:
    args = parse_args()
    seeds = (
        [args.seed]
        if args.seeds is None
        else [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    )

    # Initialize distributed for multi-host
    if args.enable_sharding:
        if not GEMMA3_AVAILABLE:
            raise RuntimeError(
                "Sharding requires Gemma 3 module. Install gemma dependencies."
            )
        jax.distributed.initialize()

    # Logging
    logger.info("=" * 60)
    logger.info("PonderTTT Training")
    logger.info("=" * 60)
    logger.info(f"Model scale: {args.model_scale}")
    logger.info(f"Action: {args.action}")
    logger.info(f"Max chunks: {args.max_chunks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Seeds: {seeds}")

    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} ({devices[0].platform})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup mesh for multi-device
    mesh = None
    data_sharding = None
    if args.enable_sharding and GEMMA3_AVAILABLE:
        sharding_config = ShardingConfig(
            dcn_data_parallelism=args.dcn_data_parallelism,
            dcn_fsdp_parallelism=args.dcn_fsdp_parallelism,
            dcn_tensor_parallelism=args.dcn_tensor_parallelism,
            ici_data_parallelism=args.ici_data_parallelism,
            ici_fsdp_parallelism=args.ici_fsdp_parallelism,
            ici_tensor_parallelism=args.ici_tensor_parallelism,
        )
        mesh = create_device_mesh(sharding_config)
        data_sharding = get_data_sharding(mesh, sharding_config)
        logger.info(f"Mesh: {mesh.shape}")
        logger.info(f"Data sharding: {data_sharding}")

    # Model configuration
    model_name = get_model_name(args.model_scale)
    is_gemma = is_gemma_model(args.model_scale)
    num_ttt_steps = action_to_steps(args.action)
    cost_multiplier = action_to_cost(args.action)

    logger.info(f"\nModel: {model_name} ({'Gemma 3' if is_gemma else 'GPT-2'})")
    logger.info(f"TTT steps: {num_ttt_steps}, Cost: {cost_multiplier}x")

    # Initialize WandB
    if args.wandb_project and WANDB_AVAILABLE and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"train_{args.model_scale}_{args.action}",
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
    examples_needed = math.ceil(args.max_chunks / max(chunks_per_seq, 1)) * batch_size

    # Create training step functions
    if args.action == "ADAPTIVE":
        train_step_fn = make_adaptive_train_step(args.ssl_weight, args.gating_threshold)
        logger.info(f"Using ADAPTIVE gating (threshold={args.gating_threshold})")
    else:
        train_step_fn = make_train_step(args.ssl_weight)
    eval_step_fn = make_eval_step()

    # Model initialization function
    def init_model(seed: int) -> tuple[TTTModel, Any]:
        logger.info(f"\nInitializing model (seed={seed})...")

        if is_gemma:
            # Gemma 3
            model, config = load_ttt_model(
                model_name=model_name,
                fast_weight_type="ttt",
                dtype=jnp.bfloat16,
                seed=seed,
                load_pretrained=args.load_pretrained,
                checkpoint_path=args.checkpoint_path,
            )
        else:
            # GPT-2
            tok_vocab_size = tokenizer.get_vocab_size()
            if args.fast_weight_type == "lora":
                from ..models import LoRAConfig

                hidden_dims = {
                    "gpt2": 768,
                    "medium": 1024,
                    "large": 1280,
                    "xl": 1600,
                }
                lora_config = LoRAConfig(
                    hidden_dim=hidden_dims[args.model_scale],
                    rank=args.lora_rank,
                    alpha=float(args.lora_rank),
                    dropout_rate=0.1,
                )
                model, config = load_ttt_model(
                    model_name=model_name,
                    fast_weight_type="lora",
                    lora_config=lora_config,
                    seed=seed,
                    load_pretrained=args.load_pretrained,
                    vocab_size=tok_vocab_size,
                )
            else:
                model, config = load_ttt_model(
                    model_name=model_name,
                    fast_weight_type="ttt",
                    seed=seed,
                    load_pretrained=args.load_pretrained,
                    vocab_size=tok_vocab_size,
                )
        return model, config

    # Initialize for stats
    model, config = init_model(seeds[0])
    total_params = count_params(model)
    trainable_state = model.get_trainable_params()
    trainable_params = sum(
        x.size for x in jax.tree.leaves(trainable_state) if hasattr(x, "size")
    )

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable (TTT): {trainable_params:,}")
    logger.info(f"Frozen: {total_params - trainable_params:,}")

    # JIT compile step functions
    if args.action == "ADAPTIVE":
        jit_train_step = train_step_fn  # Adaptive uses Python branching
    elif mesh is not None:
        jit_train_step = partial(jax.jit, static_argnames=["use_ttt"])(train_step_fn)
    else:
        jit_train_step = nnx.jit(train_step_fn, static_argnames=["use_ttt"])

    if mesh is not None:
        jit_eval_step = partial(jax.jit, static_argnames=["use_ttt"])(eval_step_fn)
    else:
        jit_eval_step = nnx.jit(eval_step_fn, static_argnames=["use_ttt"])

    # Optimizer factory
    action_steps = action_to_steps(args.action)
    effective_lr = args.learning_rate / max(action_steps, 1)

    def create_optimizer(mdl: TTTModel) -> nnx.Optimizer:
        return nnx.Optimizer(
            mdl,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(effective_lr),
            ),
            wrt=nnx.All(nnx.Param),
        )

    logger.info(f"\nOptimizer: Adam (lr={effective_lr:.2e})")

    # Training loop
    logger.info("\nStarting training...")
    seed_results = []

    for seed in seeds:
        model, config = init_model(seed)
        initial_checksum = get_base_model_checksum(model)

        if action_steps > 0:
            model.train()
        else:
            model.eval()

        optimizer = create_optimizer(model)
        start_chunk = 0

        # Resume logic
        if args.resume_from and len(seeds) == 1:
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
        remainder = start_chunk % chunks_per_batch

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

        logger.info(f"\n=== Seed {seed} ===")

        with tqdm(
            total=args.max_chunks, initial=start_chunk, desc=f"Seed {seed}"
        ) as pbar:
            while chunks_processed < args.max_chunks:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    logger.info("Data exhausted")
                    break

                num_chunks_in_batch = batch["chunks"].shape[1]

                for chunk_idx in range(num_chunks_in_batch):
                    if first_batch and chunk_idx < (remainder // batch_size):
                        continue
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
                                optimizer,
                                chunk_input_ids,
                                chunk_mask,
                                chunk_pos,
                                use_ttt=True,
                            )

                    # Check stability
                    if (
                        not jnp.isfinite(metrics["loss_ce"])
                        or metrics["loss_ce"] > 20.0
                    ):
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

                first_batch = False

        # Seed complete
        if chunks_processed > 0:
            denom = chunks_processed - start_chunk + 1e-6
            final_loss = total_loss_ce / denom
            final_ppl = math.exp(final_loss)

            seed_results.append(
                {
                    "seed": seed,
                    "chunks": chunks_processed,
                    "loss": final_loss,
                    "perplexity": final_ppl,
                    "cost": total_cost,
                }
            )

            # Verify frozen weights
            final_checksum = get_base_model_checksum(model)
            if abs(final_checksum - initial_checksum) > 1e-4:
                logger.warning("⚠️ Base model weights changed!")
            else:
                logger.info("✓ Base model frozen")

            # Save results
            results = {
                "model_scale": args.model_scale,
                "action": args.action,
                "chunks": chunks_processed,
                "loss": final_loss,
                "perplexity": final_ppl,
                "seed": seed,
            }
            results_file = (
                output_dir / f"results_{args.model_scale}_{args.action}_seed{seed}.json"
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved: {results_file}")

            save_checkpoint(
                checkpoint_dir=output_dir / "checkpoints",
                step=chunks_processed,
                state={"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                metadata=results,
            )
            wait_for_checkpoints()

    # Summary
    if seed_results and len(seed_results) > 1:
        losses = jnp.array([r["loss"] for r in seed_results])
        logger.info(
            f"\nFinal: Loss = {float(losses.mean()):.4f} ± {float(losses.std()):.4f}"
        )


if __name__ == "__main__":
    main()
