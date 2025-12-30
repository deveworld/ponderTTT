"""
Compare gating methods for adaptive Test-Time Training.

Supports:
- GPT-2 models (small, medium, large, xl)
- Gemma 3 models (1b, 4b, 12b, 27b) with TPU sharding

Methods:
    1. SKIP (Baseline): No TTT updates
    2. UPDATE_1 (Fixed): Always update
    3. Random Skip: Randomly update
    4. Oracle: Upper bound
    5. PonderTTT (Ours): Reconstruction loss gating

Usage:
    # GPT-2
    python -m ponderttt.experiments.compare_methods --model_scale small --budget 2.0

    # Gemma 3
    python -m ponderttt.experiments.compare_methods --model_scale 4b --budget 2.0

    # Gemma 3 with TPU sharding
    python -m ponderttt.experiments.compare_methods --model_scale 4b --enable_sharding
"""

import argparse
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTModel
from ..utils import cross_entropy_loss

from ..gating import (
    FixedGating,
    RandomGating,
    ReconstructionGating,
)

# Optional Gemma 3 sharding
try:
    from ..models.gemma3 import (
        ShardingConfig,
        create_device_mesh,
    )

    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================


def is_gemma_model(model_scale: str) -> bool:
    """Check if model scale is Gemma 3."""
    return model_scale in ["1b", "4b", "12b", "27b"]


def get_model_name(model_scale: str) -> str:
    """Convert model scale to model name."""
    return {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl",
        "1b": "gemma3-1b",
        "4b": "gemma3-4b",
        "12b": "gemma3-12b",
        "27b": "gemma3-27b",
    }[model_scale]


def get_default_tokenizer(model_scale: str) -> str:
    """Get default tokenizer name."""
    if is_gemma_model(model_scale):
        return "google/gemma-3-1b-it"
    return get_model_name(model_scale)


def get_seq_length(model_scale: str) -> int:
    """Get sequence length for model."""
    return 4096 if is_gemma_model(model_scale) else 1024


def budget_to_update_rate(budget: float) -> float:
    """Convert budget to update rate."""
    rate = (budget - 1.0) / 2.0
    return float(min(max(rate, 0.0), 1.0))


def permute_within_chunks(input_ids: jax.Array, seed: int) -> jax.Array:
    """Permute tokens randomly within chunk."""
    B, L = input_ids.shape
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, B)
    return jax.vmap(lambda seq, k: jax.random.permutation(k, seq))(input_ids, keys)


# =============================================================================
# JIT Step Functions
# =============================================================================


def make_step_fn(is_gemma: bool):
    """Create step function for computing losses."""

    def step_fn(model, input_ids, attention_mask, position_ids, use_ttt):
        if is_gemma:
            # Gemma 3 returns (output_dict, cache)
            out, _ = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_ttt=use_ttt,
            )
        else:
            # GPT-2 returns output_dict
            out = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_ttt=use_ttt,
            )

        loss = cross_entropy_loss(
            out["logits"][:, :-1],
            input_ids[:, 1:],
            attention_mask[:, 1:],
        )
        ttt_stats = out.get("ttt_stats", {})
        return loss, ttt_stats

    return step_fn


def get_ttt_loss_from_stats(ttt_stats: dict | None) -> tuple[float, float]:
    """Extract TTT loss values from stats dict."""
    if ttt_stats is None:
        return 0.0, 0.0

    init_loss = ttt_stats.get("ttt_loss_step_0", ttt_stats.get("ttt_loss_init", 0.0))
    final_loss = ttt_stats.get("ttt_loss_step_1", ttt_stats.get("ttt_loss_final", 0.0))

    if hasattr(init_loss, "mean"):
        init_loss = float(init_loss.mean())
    if hasattr(final_loss, "mean"):
        final_loss = float(final_loss.mean())

    return float(init_loss), float(final_loss)


# =============================================================================
# Core Evaluation Functions
# =============================================================================


def evaluate_method(
    method_name: str,
    model: TTTModel,
    data_iter,
    num_batches: int,
    gating_strategy,
    step_fn,
    seed: int = 42,
    shuffle: bool = False,
) -> pd.DataFrame:
    """Evaluate a gating method."""
    results = {
        "method": [],
        "loss": [],
        "loss_skip": [],
        "loss_update": [],
        "advantage": [],
        "decision": [],
        "ttt_loss_init": [],
        "cost": [],
    }

    jit_step = nnx.jit(step_fn, static_argnames=["use_ttt"])

    for batch_idx, batch in enumerate(
        tqdm(data_iter, total=num_batches, desc=method_name)
    ):
        if batch_idx >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        batch_size = chunks.shape[0]

        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            chunk_len = chunk_input.shape[-1]

            if shuffle:
                chunk_input = permute_within_chunks(chunk_input, seed + c_idx)

            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            # Multi-host: each host has different data (dataset sharding)
            # Just convert to JAX arrays on local devices
            chunk_input = jnp.asarray(chunk_input)
            chunk_mask = jnp.asarray(chunk_mask)
            position_ids = jnp.asarray(position_ids)

            # Compute both paths
            loss_skip, _ = jit_step(
                model, chunk_input, chunk_mask, position_ids, use_ttt=False
            )
            loss_update, ttt_stats = jit_step(
                model, chunk_input, chunk_mask, position_ids, use_ttt=True
            )

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            ttt_loss_init, _ = get_ttt_loss_from_stats(ttt_stats)
            advantage = loss_skip_val - loss_update_val

            decision_result = gating_strategy.decide(
                loss_skip=loss_skip_val,
                ttt_loss_init=ttt_loss_init,
                ttt_loss_final=0.0,
            )
            decision = "UPDATE" if decision_result.should_update else "SKIP"

            gating_strategy.update_state(
                {
                    "ttt_loss_init": ttt_loss_init,
                    "was_update": decision == "UPDATE",
                }
            )

            for _ in range(batch_size):
                results["method"].append(method_name)
                results["loss"].append(
                    loss_update_val if decision == "UPDATE" else loss_skip_val
                )
                results["loss_skip"].append(loss_skip_val)
                results["loss_update"].append(loss_update_val)
                results["advantage"].append(advantage)
                results["decision"].append(decision)
                results["ttt_loss_init"].append(ttt_loss_init)
                results["cost"].append(3.0 if decision == "UPDATE" else 1.0)

    return pd.DataFrame(results)


def evaluate_oracle(
    model: TTTModel,
    data_iter,
    num_batches: int,
    target_update_rate: float,
    step_fn,
    seed: int = 42,
    shuffle: bool = False,
) -> pd.DataFrame:
    """Oracle: select top-k% chunks by advantage."""
    chunk_data = []
    jit_step = nnx.jit(step_fn, static_argnames=["use_ttt"])

    for batch_idx, batch in enumerate(
        tqdm(data_iter, total=num_batches, desc="Oracle")
    ):
        if batch_idx >= num_batches:
            break

        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        batch_size = chunks.shape[0]

        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            chunk_len = chunk_input.shape[-1]

            if shuffle:
                chunk_input = permute_within_chunks(chunk_input, seed + c_idx)

            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            # Multi-host: each host has different data
            chunk_input = jnp.asarray(chunk_input)
            chunk_mask = jnp.asarray(chunk_mask)
            position_ids = jnp.asarray(position_ids)

            loss_skip, _ = jit_step(
                model, chunk_input, chunk_mask, position_ids, use_ttt=False
            )
            loss_update, ttt_stats = jit_step(
                model, chunk_input, chunk_mask, position_ids, use_ttt=True
            )

            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            ttt_loss_init, _ = get_ttt_loss_from_stats(ttt_stats)

            chunk_data.append(
                {
                    "batch_idx": batch_idx,
                    "chunk_idx": c_idx,
                    "batch_size": batch_size,
                    "loss_skip": loss_skip_val,
                    "loss_update": loss_update_val,
                    "advantage": loss_skip_val - loss_update_val,
                    "ttt_loss_init": ttt_loss_init,
                }
            )

    # Select top-k
    num_to_update = max(1, int(len(chunk_data) * target_update_rate))
    sorted_chunks = sorted(chunk_data, key=lambda x: x["advantage"], reverse=True)
    update_set = set(
        (c["batch_idx"], c["chunk_idx"]) for c in sorted_chunks[:num_to_update]
    )

    results = {
        "method": [],
        "loss": [],
        "loss_skip": [],
        "loss_update": [],
        "advantage": [],
        "decision": [],
        "ttt_loss_init": [],
        "cost": [],
    }

    for c in chunk_data:
        decision = (
            "UPDATE" if (c["batch_idx"], c["chunk_idx"]) in update_set else "SKIP"
        )
        for _ in range(c["batch_size"]):
            results["method"].append("Oracle")
            results["loss"].append(
                c["loss_update"] if decision == "UPDATE" else c["loss_skip"]
            )
            results["loss_skip"].append(c["loss_skip"])
            results["loss_update"].append(c["loss_update"])
            results["advantage"].append(c["advantage"])
            results["decision"].append(decision)
            results["ttt_loss_init"].append(c["ttt_loss_init"])
            results["cost"].append(3.0 if decision == "UPDATE" else 1.0)

    return pd.DataFrame(results)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Compare gating methods")

    # Model
    parser.add_argument(
        "--model_scale",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl", "1b", "4b", "12b", "27b"],
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # Experiment
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--num_eval_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--language", type=str, default="Python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--skip_examples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--methods", type=str, default="all")

    # Sharding
    parser.add_argument("--enable_sharding", action="store_true")
    parser.add_argument("--dcn_data_parallelism", type=int, default=-1)
    parser.add_argument("--dcn_fsdp_parallelism", type=int, default=1)
    parser.add_argument("--dcn_tensor_parallelism", type=int, default=1)
    parser.add_argument("--ici_data_parallelism", type=int, default=1)
    parser.add_argument("--ici_fsdp_parallelism", type=int, default=-1)
    parser.add_argument("--ici_tensor_parallelism", type=int, default=1)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    is_gemma = is_gemma_model(args.model_scale)

    # Initialize distributed for multi-host
    if args.enable_sharding:
        if not GEMMA3_AVAILABLE:
            raise RuntimeError("Sharding requires Gemma 3 module")
        jax.distributed.initialize()

    logger.info("=" * 60)
    logger.info("PonderTTT Method Comparison")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_scale} ({'Gemma 3' if is_gemma else 'GPT-2'})")
    logger.info(f"Budget: {args.budget}")
    logger.info(f"Batches: {args.num_eval_batches}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Devices: {len(jax.devices())}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup sharding (mesh used for distributed initialization)
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
        logger.info(f"Mesh: {mesh.shape}")

    # Load model
    model_name = get_model_name(args.model_scale)
    tokenizer_name = get_default_tokenizer(args.model_scale)
    tokenizer = get_tokenizer(tokenizer_name)
    seq_length = get_seq_length(args.model_scale)

    if is_gemma:
        model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            dtype=jnp.bfloat16,
            seed=args.seed,
            load_pretrained=True,
            checkpoint_path=args.checkpoint_path,
        )
    else:
        model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=tokenizer.get_vocab_size(),
            checkpoint_path=args.checkpoint_path,
        )
    model.eval()

    target_update_rate = budget_to_update_rate(args.budget)
    logger.info(f"Target update rate: {target_update_rate:.1%}")

    # Parse methods
    methods_str = args.methods.lower()
    methods = (
        ["skip", "update1", "random", "oracle", "ours"]
        if methods_str == "all"
        else [m.strip() for m in methods_str.split(",")]
    )

    # Create step function
    step_fn = make_step_fn(is_gemma)

    # Data iterator factory
    def make_data_iter():
        return create_data_iterator(
            tokenizer=tokenizer,
            split=args.split,
            batch_size=args.batch_size,
            seq_length=seq_length,
            chunk_size=512,
            max_examples=args.num_eval_batches * args.batch_size * 2,
            num_workers=args.num_workers,
            language=args.language,
            skip_examples=args.skip_examples,
        )

    all_results = []

    # Evaluate methods
    if "skip" in methods:
        logger.info("\n--- SKIP ---")
        gating = FixedGating(always_update=False)
        df = evaluate_method(
            "SKIP",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            step_fn,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        logger.info(f"  Avg loss: {df['loss'].mean():.4f}")

    if "update1" in methods:
        logger.info("\n--- UPDATE_1 ---")
        gating = FixedGating(always_update=True)
        df = evaluate_method(
            "UPDATE_1",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            step_fn,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        logger.info(f"  Avg loss: {df['loss'].mean():.4f}")

    if "random" in methods:
        logger.info("\n--- Random Skip ---")
        gating = RandomGating(update_prob=target_update_rate)
        df = evaluate_method(
            "Random Skip",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            step_fn,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        logger.info(
            f"  Avg loss: {df['loss'].mean():.4f}, Update rate: {(df['decision'] == 'UPDATE').mean():.1%}"
        )

    if "oracle" in methods:
        logger.info("\n--- Oracle ---")
        df = evaluate_oracle(
            model,
            make_data_iter(),
            args.num_eval_batches,
            target_update_rate,
            step_fn,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        logger.info(
            f"  Avg loss: {df['loss'].mean():.4f}, Update rate: {(df['decision'] == 'UPDATE').mean():.1%}"
        )

    if "ours" in methods:
        logger.info("\n--- PonderTTT (Ours) ---")
        gating = ReconstructionGating(target_rate=target_update_rate)
        df = evaluate_method(
            "Ours",
            model,
            make_data_iter(),
            args.num_eval_batches,
            gating,
            step_fn,
            args.seed,
            args.shuffle,
        )
        all_results.append(df)
        logger.info(
            f"  Avg loss: {df['loss'].mean():.4f}, Update rate: {(df['decision'] == 'UPDATE').mean():.1%}"
        )

    # Summary
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        summary = (
            combined_df.groupby("method").agg({"loss": "mean", "cost": "mean"}).round(4)
        )
        logger.info(f"\n{summary}")

        combined_df.to_csv(output_dir / "results.csv", index=False)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "model_scale": args.model_scale,
                    "budget": args.budget,
                    "methods": summary.to_dict(),
                },
                f,
                indent=2,
            )

        logger.info(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
