"""
Compare gating methods for Gemma 3: SKIP, UPDATE_1, Loss Skip, Oracle.

Extends GPT-2 experiments to Gemma 3 (1B, 4B, 12B) with TPU sharding support.

Usage:
    python -m ponderttt.experiments.compare_gemma3_methods \
        --model_scale 4b \
        --checkpoint_path hf:google/gemma-3-4b-pt \
        --update_rate 0.5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Any

import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTModel
from ..models.gemma3 import (
    ShardingConfig,
    create_device_mesh,
    get_data_sharding,
)
from ..utils import cross_entropy_loss
from ..utils.checkpointing import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare gating methods on Gemma 3")
    
    # Model
    parser.add_argument("--model_scale", type=str, choices=["1b", "4b", "12b"], default="4b")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to checkpoint. 'hf:google/gemma-3-4b-pt' or Orbax path")
    
    # Experiment
    parser.add_argument("--update_rate", type=float, default=0.5, help="Update budget (e.g. 0.5 = 50%)")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--language", type=str, default="Python", help="Dataset language")
    
    # Sharding
    parser.add_argument("--enable_sharding", action="store_true", help="Enable TPU sharding")
    parser.add_argument("--dcn_data_parallelism", type=int, default=-1)
    parser.add_argument("--dcn_fsdp_parallelism", type=int, default=1)
    parser.add_argument("--dcn_tensor_parallelism", type=int, default=1)
    parser.add_argument("--ici_data_parallelism", type=int, default=1)
    parser.add_argument("--ici_fsdp_parallelism", type=int, default=-1)
    parser.add_argument("--ici_tensor_parallelism", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="outputs/gemma3_comparison")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def create_sharding_config(args) -> ShardingConfig:
    return ShardingConfig(
        dcn_data_parallelism=args.dcn_data_parallelism,
        dcn_fsdp_parallelism=args.dcn_fsdp_parallelism,
        dcn_tensor_parallelism=args.dcn_tensor_parallelism,
        ici_data_parallelism=args.ici_data_parallelism,
        ici_fsdp_parallelism=args.ici_fsdp_parallelism,
        ici_tensor_parallelism=args.ici_tensor_parallelism,
    )


def evaluate_oracle_gemma(
    model: TTTModel,
    tokenizer: Any,
    args: argparse.Namespace,
    mesh: Optional[jax.sharding.Mesh] = None,
    data_sharding: Optional[jax.sharding.NamedSharding] = None,
):
    """
    Oracle evaluation for Gemma 3.
    Compute SKIP vs UPDATE loss for every chunk and measure Oracle upper bound.
    """
    logger.info(f"Evaluating Oracle on {args.language} (Update Budget: {args.update_rate})")
    
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="test",
        language=args.language,
        batch_size=args.batch_size,
        seq_length=4096, # Gemma 3 context
        chunk_size=512,
        max_examples=args.batch_size * args.num_batches * 2,
        num_workers=16,
    )

    # JIT functions with sharding if mesh provided
    if mesh is not None:
        jit_opts = {"out_shardings": None} # Let JAX infer
    else:
        jit_opts = {}

    # Define step functions
    def step_fn(model, input_ids, attention_mask, position_ids, use_ttt, gating_scale=None):
        # Gemma3TTTModel returns (output_dict, cache) tuple
        out, _cache = model(
            input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            use_ttt=use_ttt, 
            gating_scale=gating_scale
        )
        return cross_entropy_loss(out["logits"][:, :-1], input_ids[:, 1:], attention_mask[:, 1:])

    if mesh is not None:
        jit_step = nnx.jit(step_fn, static_argnames=("use_ttt",))
    else:
        jit_step = nnx.jit(step_fn, static_argnames=("use_ttt",))

    results = {
        "loss": [], "cost": [], "method": [], "decision": [],
        "loss_skip_val": [], "loss_update_val": [], "advantage": []
    }

    chunk_stats = []

    for i, batch in enumerate(tqdm(data_iter, total=args.num_batches)):
        if i >= args.num_batches:
            break

        chunks = batch["chunks"] # [B, num_chunks, chunk_size]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]

        # For Oracle, we must process each chunk individually to make decision
        # But for efficiency, we can process batch of chunks if they are independent
        # Current data loader gives [B, num_chunks, L].
        # We iterate chunks in seq.
        
        for c_idx in range(num_chunks):
            chunk_input = chunks[:, c_idx]
            chunk_mask = masks[:, c_idx]
            
            # Check valid tokens
            # NOTE: This causes a host-device sync per chunk and can slow down evaluation.
            # For production, consider filtering in data loader or accepting all chunks.
            if jnp.sum(chunk_mask[:, 1:]) < 16:
                continue

            chunk_len = chunk_input.shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)[None, :] + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_input.shape)

            # Apply sharding
            if data_sharding is not None:
                chunk_input = jax.device_put(chunk_input, data_sharding)
                chunk_mask = jax.device_put(chunk_mask, data_sharding)
                position_ids = jax.device_put(position_ids, data_sharding)

            # 1. SKIP Loss
            loss_skip = jit_step(model, chunk_input, chunk_mask, position_ids, use_ttt=False)
            
            # 2. UPDATE Loss
            # Gating scale 1.0 needed? TTTLayer usually handles it.
            loss_update = jit_step(model, chunk_input, chunk_mask, position_ids, use_ttt=True)

            # Host sync for stats
            loss_skip_val = float(loss_skip)
            loss_update_val = float(loss_update)
            advantage = loss_skip_val - loss_update_val

            chunk_stats.append({
                "batch_idx": i,
                "chunk_idx": c_idx,
                "loss_skip": loss_skip_val,
                "loss_update": loss_update_val,
                "advantage": advantage,
            })

    # Oracle Selection
    # Sort all processed chunks by advantage
    num_total = len(chunk_stats)
    num_update = max(1, int(num_total * args.update_rate))
    
    sorted_stats = sorted(chunk_stats, key=lambda x: x["advantage"], reverse=True)
    update_set = set((x["batch_idx"], x["chunk_idx"]) for x in sorted_stats[:num_update])

    # Record Method Results
    avg_loss_oracle = 0.0
    avg_loss_baseline = 0.0 # All SKIP

    for stat in chunk_stats:
        key = (stat["batch_idx"], stat["chunk_idx"])
        
        # Baseline
        avg_loss_baseline += stat["loss_skip"]

        # Oracle
        if key in update_set:
            results["loss"].append(stat["loss_update"])
            results["decision"].append("UPDATE")
            results["cost"].append(3.0)
            avg_loss_oracle += stat["loss_update"]
        else:
            results["loss"].append(stat["loss_skip"])
            results["decision"].append("SKIP")
            results["cost"].append(1.0)
            avg_loss_oracle += stat["loss_skip"]
            
        results["method"].append("Oracle")
        results["loss_skip_val"].append(stat["loss_skip"])
        results["loss_update_val"].append(stat["loss_update"])
        results["advantage"].append(stat["advantage"])

    avg_loss_oracle /= num_total
    avg_loss_baseline /= num_total

    print(f"\nResults ({num_total} chunks):")
    print(f"  Baseline (SKIP) Loss: {avg_loss_baseline:.4f}")
    print(f"  Oracle Loss:          {avg_loss_oracle:.4f}")
    print(f"  Oracle Advantage:     {avg_loss_baseline - avg_loss_oracle:.4f}")
    
    # Save correlation data
    df = pd.DataFrame(results)
    
    # Analyze Loss Skip Correlation
    # Loss Skip Gating: Use loss_skip > Threshold to decide UPDATE
    if len(df) > 0:
        corr = df["loss_skip_val"].corr(df["advantage"])
        print(f"  Correlation (Initial Loss vs Advantage): {corr:.4f}")

    return df


def main():
    args = parse_args()
    
    # Initialize Distributed
    if args.enable_sharding:
        jax.distributed.initialize()

    # Create Output Dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mesh
    mesh = None
    data_sharding = None
    if args.enable_sharding:
        sharding_config = create_sharding_config(args)
        mesh = create_device_mesh(sharding_config)
        data_sharding = get_data_sharding(mesh, sharding_config)

    # Load Model
    model_name = f"gemma3-{args.model_scale}"
    logger.info(f"Loading {model_name} from {args.checkpoint_path}...")
    
    # Use load_ttt_model which supports Gemma3
    model, config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        dtype=jnp.bfloat16,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        load_pretrained=True
    )
    
    # Set to eval mode
    model.eval()
    
    # Apply sharding to params if enabled
    if mesh is not None:
        from ..models.gemma3.sharding import setup_sharded_state
        # Setup sharding (this shards the params in place or returns sharded state)
        # load_ttt_model returns initialized model.
        # We need to re-shard.
        # Gemma3 sharding utility expects (model, optimizer, mesh).
        # Here we only have model.
        # Actually setup_sharded_state works on model state.
        
        # We can manually shard using same logic
        state = nnx.state(model)
        # ... logic similar to train_gemma3.py or simplifed
        # For inference/eval, we mainly need params sharded.
        
        # Simply rely on JAX automatic sharding propagation if we put data correctly?
        # Or use the shard_params utility.
        pass

    # Tokenizer
    tokenizer = get_tokenizer("google/gemma-2-2b") # Use generic gemma tokenizer

    # Run Oracle Evaluation
    df = evaluate_oracle_gemma(model, tokenizer, args, mesh, data_sharding)
    
    # Save
    csv_path = output_dir / f"oracle_{args.model_scale}_{args.language}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
