"""
Compare Differentiable Gating vs RL Baseline (PPO).

Usage:
    python -m ponderttt.experiments.compare_methods --model_scale 125m --budget 2.0
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flax import nnx
from tqdm import tqdm

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, TTTConfig, PolicyConfig, PolicyNetwork
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Compare optimization methods")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m", "1b"])
    parser.add_argument("--budget", type=float, default=2.0, help="Target budget (avg steps)")
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--diff_checkpoint", type=str, help="Path to differentiable gating checkpoint (optional)")
    parser.add_argument("--rl_checkpoint", type=str, help="Path to RL policy checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_model(
    method_name: str,
    model_scale: str,
    budget_target: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    gating_net: nnx.Module = None,  # GatingNetwork or PolicyNetwork
    is_rl: bool = False,
):
    print(f"\nEvaluating {method_name}...")
    
    # Load Model & Data
    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    tokenizer = get_tokenizer(model_name)
    
    ttt_model, _ = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )
    
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )
    
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",  # Use validation split in real scenario
        batch_size=batch_size,
        seq_length=2048,
        chunk_size=512,
        max_examples=batch_size * num_batches * 2,
    )
    
    results = {
        "loss": [],
        "cost": [],
        "method": []
    }
    
    step_map = [0, 1, 2, 4]
    costs_map = [1.0, 3.0, 6.0, 12.0]
    
    # Evaluation Loop
    for i, batch in enumerate(tqdm(data_iter, total=num_batches, desc=method_name)):
        if i >= num_batches:
            break
            
        chunks = batch["chunks"]
        masks = batch["chunk_attention_mask"]
        num_chunks = chunks.shape[1]
        total_budget = budget_target * num_chunks
        current_spend = 0.0
        
        feature_extractor.reset_history()
        
        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx]
            }
            
            # Budget Feature
            budget_rem = max(0.0, (total_budget - current_spend) / total_budget) if total_budget > 0 else 0.0
            
            # Extract Features (using base forward)
            out_base = ttt_model(chunk_batch["input_ids"], use_ttt=False)
            features = feature_extractor.extract(
                input_ids=chunk_batch["input_ids"],
                logits=out_base["logits"],
                attention_mask=chunk_batch["attention_mask"],
                budget_remaining=budget_rem,
            )
            
            # Decision
            if gating_net is None:
                # Fixed Baseline (e.g. Skip) or Oracle
                # Let's implement Fixed Baseline: Always use budget_target roughly
                # E.g. if budget=2.0, use UPDATE_1 (cost 3) mixed with SKIP? 
                # For simplicity, "Baseline" = Fixed UPDATE_1 (Cost 3) if budget >= 3, else SKIP?
                # Or better: "Random" policy respecting budget?
                # Let's assume this function is called with trained networks.
                # If None, run SKIP baseline.
                scale = 0.0
                cost = 1.0
            elif is_rl:
                # RL Policy
                # Policy outputs action index
                out = gating_net(features, deterministic=True)
                action = int(out["action"][0]) # Assume batch consistency or take first
                steps = step_map[action]
                cost = costs_map[action]
                scale = float(steps) # Not used for continuous scaling but for logging
                
                # Run TTT with steps (Simulated by calling TTTLayer iteratively or modifying it)
                # Our TTTLayer supports `gating_scale`.
                # For discrete steps, we can map 0->0, 1->1, 2->2, 4->4.
                # But TTTLayer is "Continuous" implementation now?
                # Actually `gating_scale` multiplies `eta`.
                # If we want Discrete Update, we set `gating_scale` such that it mimics K steps?
                # No, `gating_scale` scales the LR of ONE step.
                # To support K steps, we need the loop in `train_policy` or `TTTLayer`.
                # For comparison fairness, let's assume Differentiable approach uses `gating_scale` (single pass),
                # and RL approach uses Multi-step (iterative).
                # This puts Differentiable at disadvantage? No, Continuous is often better.
                # Let's standardize: BOTH use "Single Pass with Scaled LR" for this evaluation?
                # OR: We use the `action_steps` for RL to actually run loop.
                pass # Implemented below
            else:
                # Differentiable Gating
                # Outputs scalar scale
                scale = float(gating_net(features, train=False)[0, 0])
                cost = 1.0 + scale * 0.5 # Approximate cost model? 
                # Real cost model: 
                # TTT cost = 1 (fwd) + 2 (bwd) * scale? 
                # No, Diff TTT is constant cost (1 fwd + 1 bwd + 1 fwd) = 3x ?
                # Wait. "Differentiable TTT" is usually 1-step unrolled.
                # So cost is Fixed at ~3x (Update_1).
                # Scaling `eta` doesn't change FLOPs.
                # So Diff TTT is always Cost=3.0?
                # If gate=0, we can skip.
                # If we implement dynamic skip:
                if scale < 0.1:
                    cost = 1.0 # Skip
                else:
                    cost = 3.0 # 1-step update
                
            
            # Execution
            if is_rl:
                # RL: Discrete Steps
                # Run K updates
                # We need to manually run TTT update K times or use a helper.
                # For simplicity in comparison script, let's use `ttt_model` with `use_ttt=True` (1 step)
                # looped K times? 
                # `ttt_model` doesn't support external loop easily without state management.
                # Let's assume RL chooses "Scale" from {0, 1, 2, 4} and we apply it as `gating_scale`.
                # This makes comparison fair: "Discrete Scale" vs "Continuous Scale".
                
                # If RL chooses 4, we use scale=4.0 in one step.
                # (Approximation: 4 small steps ~= 1 big step with 4x LR)
                real_scale = float(costs_map[step_map.index(int(scale))] if scale in step_map else scale)
                
                out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=True, gating_scale=jnp.array([[scale]]))
                loss = float(cross_entropy_loss(out_ttt["logits"][:, :-1], chunk_batch["input_ids"][:, 1:], chunk_batch["attention_mask"][:, 1:]))
                
            else:
                # Differentiable: Continuous Scale
                # If scale close to 0, use SKIP (cost 1)
                if scale < 0.01:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=False)
                    cost = 1.0
                else:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=True, gating_scale=jnp.array([[scale]]))
                    cost = 3.0 # Fixed cost for 1-step
                
                loss = float(cross_entropy_loss(out_ttt["logits"][:, :-1], chunk_batch["input_ids"][:, 1:], chunk_batch["attention_mask"][:, 1:]))
            
            results["loss"].append(loss)
            results["cost"].append(cost)
            results["method"].append(method_name)
            
            current_spend += cost
            feature_extractor.update_history(loss, cost)

    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    print("="*60)
    print("Comparison: Differentiable Gating vs RL vs Baseline")
    print("="*60)
    
    # 1. Initialize/Load Networks
    rngs = nnx.Rngs(args.seed)
    
    # Differentiable Network (Random Init if no checkpoint)
    diff_net = GatingNetwork(
        config=GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0), 
        rngs=rngs
    )
    # TODO: Load checkpoint if args.diff_checkpoint
    
    # RL Network (Random Init if no checkpoint)
    rl_net = PolicyNetwork(
        config=PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4),
        rngs=rngs
    )
    # TODO: Load checkpoint if args.rl_checkpoint
    
    all_results = []
    
    # 2. Evaluate Baselines
    df_skip = evaluate_model("SKIP (Baseline)", args.model_scale, 0.0, args.num_eval_batches, args.batch_size, args.seed, None)
    all_results.append(df_skip)
    
    # 3. Evaluate Differentiable
    df_diff = evaluate_model("Differentiable", args.model_scale, args.budget, args.num_eval_batches, args.batch_size, args.seed, diff_net, is_rl=False)
    all_results.append(df_diff)
    
    # 4. Evaluate RL
    df_rl = evaluate_model("RL (PPO)", args.model_scale, args.budget, args.num_eval_batches, args.batch_size, args.seed, rl_net, is_rl=True)
    all_results.append(df_rl)
    
    # 5. Visualize & Report
    full_df = pd.concat(all_results)
    
    print("\n=== Final Results ===")
    summary = full_df.groupby("method").agg({"loss": "mean", "cost": "mean"}).sort_values("loss")
    print(summary)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary.to_csv(output_path / "summary.csv")
    full_df.to_csv(output_path / "detailed_results.csv")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x="cost", y="loss", hue="method", alpha=0.6)
    plt.axhline(y=summary.loc["SKIP (Baseline)", "loss"], color='gray', linestyle='--', label="Baseline Loss")
    plt.title(f"Cost-Quality Tradeoff (Budget ~{args.budget}x)")
    plt.savefig(output_path / "tradeoff_plot.png")
    print(f"\nPlots saved to {output_path}")

if __name__ == "__main__":
    main()
