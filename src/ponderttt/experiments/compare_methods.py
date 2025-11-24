"""
Compare Differentiable Gating vs RL Baseline (PPO).

Usage:
    python -m ponderttt.experiments.compare_methods --model_scale 125m --budget 2.0
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import optax
from flax import nnx
from tqdm import tqdm
from typing import Optional, cast

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, PolicyConfig, PolicyNetwork, TTTTransformerLM
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import load_checkpoint


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
    parser.add_argument("--language", type=str, default="Python", help="Programming language for OOD testing")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train/validation/test)")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers for data downloading")
    return parser.parse_args()


def evaluate_model(
    method_name: str,
    model_scale: str,
    budget_target: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    gating_net: Optional[GatingNetwork | PolicyNetwork] = None,
    is_rl: bool = False,
    model: Optional[TTTTransformerLM] = None,  # Accept pre-loaded model
    language: str = "Python",
    split: str = "test",
    num_workers: int = 32,
):
    print(f"\nEvaluating {method_name} on {language} ({split})...")
    
    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    tokenizer = get_tokenizer(model_name)
    
    # Load TTT Model if not provided
    if model is None:
        ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=seed,
            load_pretrained=True,
            vocab_size=tokenizer.get_vocab_size(),
        )
    else:
        ttt_model = model
    
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )
    
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split=split,
        language=language,
        batch_size=batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=batch_size * num_batches * 2,
        num_workers=num_workers,
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
            scale = 0.0
            cost = 1.0
            loss = 0.0

            if gating_net is None:
                # Fixed Baseline (e.g. Skip) 
                scale = 0.0
                cost = 1.0
                
                # If using standard SKIP, we just use out_base
                loss = float(cross_entropy_loss(out_base["logits"][:, :-1], chunk_batch["input_ids"][:, 1:], chunk_batch["attention_mask"][:, 1:]))
                
            elif is_rl and isinstance(gating_net, PolicyNetwork):
                # RL Policy
                out = gating_net(features, deterministic=True)
                action = int(out["action"][0]) 
                steps = step_map[action]
                cost = costs_map[action]
                scale = float(steps) 
                
                # For discrete RL, we simulate K steps or use scaling approximation
                # Assuming scaling approx for fairness in this script
                if steps == 0:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=False)
                else:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=True, gating_scale=jnp.array([[scale]]))
                
                loss = float(cross_entropy_loss(out_ttt["logits"][:, :-1], chunk_batch["input_ids"][:, 1:], chunk_batch["attention_mask"][:, 1:]))
                
            elif isinstance(gating_net, GatingNetwork):
                # Differentiable Gating
                scale = float(gating_net(features, train=False)[0, 0])
                
                if scale < 0.01:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=False)
                    cost = 1.0
                else:
                    out_ttt = ttt_model(chunk_batch["input_ids"], use_ttt=True, gating_scale=jnp.array([[scale]]))
                    # Cost model (1 step update)
                    cost = 3.0 
                
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
    
    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale]
    tokenizer = get_tokenizer(model_name)
    vocab_size = tokenizer.get_vocab_size()

    # Differentiable Network
    diff_net = GatingNetwork(
        config=GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0), 
        rngs=rngs
    )
    
    diff_ttt_model = None
    
    if args.diff_checkpoint:
        print(f"Loading Differentiable Gating checkpoint from {args.diff_checkpoint}...")
        
        # Instantiate fresh TTT model to load weights into
        diff_ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=vocab_size,
        )
        
        class TrainableSystem(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None
        
        # Reconstruct system and update state
        trainable_system = TrainableSystem(diff_ttt_model, diff_net)
        
        # Create target for restoration to preserve NNX State structure
        # We must match the saved structure exactly, including metadata, to avoid Orbax errors.
        # Note: We only load the model state, ignoring the optimizer state
        target = {
            "state": {"model": nnx.state(trainable_system)},
            "step": 0,
            "metadata": {
                "model_scale": "",
                "max_steps": 0.0,
                "budget_limit": 0.0,
            }
        }

        ckpt = load_checkpoint(args.diff_checkpoint, target=target)
        nnx.update(trainable_system, ckpt["state"]["model"])
        print("Differentiable Gating and TTT weights loaded.")

    # RL Network
    rl_net = PolicyNetwork(
        config=PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4),
        rngs=rngs
    )
    if args.rl_checkpoint:
        print(f"Loading RL Policy checkpoint from {args.rl_checkpoint}...")
        # Note: We only load the policy state, ignoring the optimizer state
        target = {
            "state": {"policy": nnx.state(rl_net)},
            "step": 0,
            "metadata": {
                "seed": 0,
                "model_scale": "",
                "budget_limit": 0.0,
            }
        }
        ckpt = load_checkpoint(args.rl_checkpoint, target=target)
        nnx.update(rl_net, ckpt["state"]["policy"])
        print("RL Policy weights loaded.")
    
    all_results = []
    
    # 2. Evaluate Baselines (SKIP)
    # Use fresh model for baseline
    df_skip = evaluate_model(
        "SKIP (Baseline)", 
        args.model_scale, 
        0.0, 
        args.num_eval_batches, 
        args.batch_size, 
        args.seed, 
        None,
        is_rl=False,
        language=args.language,
        split=args.split,
        num_workers=args.num_workers,
    )
    all_results.append(df_skip)
    
    # 3. Evaluate Differentiable
    # Use loaded model if available
    df_diff = evaluate_model(
        "Differentiable", 
        args.model_scale, 
        args.budget, 
        args.num_eval_batches, 
        args.batch_size, 
        args.seed, 
        diff_net, 
        is_rl=False,
        model=diff_ttt_model, # Pass the loaded model
        language=args.language,
        split=args.split,
        num_workers=args.num_workers,
    )
    all_results.append(df_diff)
    
    # 4. Evaluate RL
    # RL uses standard model (policy chooses actions)
    df_rl = evaluate_model(
        "RL (PPO)", 
        args.model_scale, 
        args.budget, 
        args.num_eval_batches, 
        args.batch_size, 
        args.seed, 
        rl_net, 
        is_rl=True,
        language=args.language,
        split=args.split,
        num_workers=args.num_workers,
    )
    all_results.append(df_rl)
    
    # 5. Visualize & Report
    full_df = pd.concat(all_results)
    
    print("\n=== Final Results ===")
    summary = cast(pd.DataFrame, full_df.groupby("method").agg({"loss": "mean", "cost": "mean"})).sort_values(by="loss")
    print(summary)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary.to_csv(output_path / "summary.csv")
    full_df.to_csv(output_path / "detailed_results.csv")
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x="cost", y="loss", hue="method", alpha=0.6)
    if "SKIP (Baseline)" in summary.index:
        baseline_loss = cast(float, summary.loc["SKIP (Baseline)", "loss"])
        plt.axhline(y=baseline_loss, color='gray', linestyle='--', label="Baseline Loss")
    plt.title(f"Cost-Quality Tradeoff (Budget ~{args.budget}x)")
    plt.savefig(output_path / "tradeoff_plot.png")
    print(f"\nPlots saved to {output_path}")

if __name__ == "__main__":
    main()