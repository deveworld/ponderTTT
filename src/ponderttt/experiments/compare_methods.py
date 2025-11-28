"""
Compare Differentiable Gating vs RL Baseline (PPO).

Usage:
    python -m ponderttt.experiments.compare_methods --model_scale 125m --budget 2.0
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flax import nnx
import optax
from tqdm import tqdm
from typing import Optional, cast

from ..data import create_data_iterator, get_tokenizer
from ..models import load_ttt_model, PolicyConfig, PolicyNetwork, TTTTransformerLM
from ..models.gating_nnx import GatingConfig, GatingNetwork, BinaryGatingConfig, BinaryGatingNetwork
from .training_utils import run_chunk_step
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import load_checkpoint


def unwrap_state(state):
    """Recursively unwrap Orbax-serialized NNX state dicts (remove 'value' wrappers)."""
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Compare optimization methods")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m", "1b"])
    parser.add_argument("--budget", type=float, default=2.0, help="Target budget (avg steps)")
    parser.add_argument("--num_eval_batches", type=int, default=20, help="Number of batches for evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (must be 1 for dynamic gating evaluation)")
    parser.add_argument("--diff_checkpoint", type=str, help="Path to differentiable gating checkpoint (optional)")
    parser.add_argument("--rl_checkpoint", type=str, help="Path to RL policy checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", type=str, default="Python", help="Programming language for OOD testing")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/validation/test). Note: The Stack v2 only has 'train'.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers for data downloading")
    parser.add_argument("--rl_learning_rate", type=float, default=5e-5, help="Learning rate for RL TTT updates during evaluation")
    parser.add_argument("--rl_max_grad_norm", type=float, default=1.0, help="Gradient norm clip for RL evaluation updates")
    parser.add_argument("--rl_ssl_weight", type=float, default=0.1, help="Auxiliary SSL loss weight when replaying RL updates")
    parser.add_argument("--hard_skip_threshold", type=float, default=0.1, help="Hard Skip threshold: skip TTT if gating scale < threshold (default: 0.1)")
    parser.add_argument("--binary_gating_checkpoint", type=str, help="Path to Binary Gating (Hard Skip) checkpoint (optional)")
    return parser.parse_args()


def evaluate_model(
    method_name: str,
    model_scale: str,
    budget_target: float,
    num_batches: int,
    batch_size: int,
    seed: int,
    gating_net: Optional[GatingNetwork | BinaryGatingNetwork | PolicyNetwork] = None,
    is_rl: bool = False,
    model: Optional[TTTTransformerLM] = None,  # Accept pre-loaded model
    language: str = "Python",
    split: str = "test",
    num_workers: int = 32,
    rl_learning_rate: float = 5e-5,
    rl_max_grad_norm: float = 1.0,
    rl_ssl_weight: float = 0.1,
    hard_skip_threshold: float = 0.1,  # Hard Skip: skip TTT if scale < threshold (for GatingNetwork)
):
    if gating_net is not None:
        assert batch_size == 1, "Batch size must be 1 for dynamic gating evaluation (mixed SKIP/TTT strategies)."

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

    def clone_state(state):
        if state is None:
            return None
        return jax.tree_util.tree_map(
            lambda x: x.copy() if hasattr(x, "copy") else jnp.array(x),
            state,
        )

    fast_layer_init = None
    fast_norm_init = None

    if is_rl:
        ttt_model.train()
        fast_layer_init = clone_state(nnx.state(ttt_model.fast_layer))
        fast_norm_init = clone_state(nnx.state(ttt_model.fast_norm)) if hasattr(ttt_model, "fast_norm") else None

    def reset_rl_state():
        if not is_rl: 
            return
        nnx.update(ttt_model.fast_layer, clone_state(fast_layer_init))
        if fast_norm_init is not None:
            nnx.update(ttt_model.fast_norm, clone_state(fast_norm_init))

    def create_rl_optimizer():
        if not is_rl: 
            return
        return nnx.Optimizer(
            ttt_model,
            optax.chain(
                optax.clip_by_global_norm(rl_max_grad_norm),
                optax.adam(rl_learning_rate),
            ),
            wrt=nnx.All(nnx.Param),
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
    # Cost model: 1 (base forward) + 2 * num_steps
    # SKIP=1, UPDATE_1=3, UPDATE_2=5, UPDATE_4=9
    costs_map = [1.0, 3.0, 5.0, 9.0]
    
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

        if is_rl:
            reset_rl_state()
            rl_optimizer = create_rl_optimizer()
        else:
            rl_optimizer = None
        
        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx]
            }

            chunk_len = chunk_batch["input_ids"].shape[-1]
            position_ids = jnp.arange(chunk_len, dtype=jnp.int32)
            position_ids = position_ids + c_idx * chunk_len
            position_ids = jnp.broadcast_to(position_ids, chunk_batch["input_ids"].shape)
            chunk_batch_with_pos = {
                **chunk_batch,
                "position_ids": position_ids,
            }
            
            # Budget Feature
            budget_rem = (total_budget - current_spend) / total_budget if total_budget > 0 else 0.0
            
            # Extract Features (using base forward)
            out_base = ttt_model(
                chunk_batch["input_ids"],
                attention_mask=chunk_batch["attention_mask"],
                position_ids=chunk_batch_with_pos["position_ids"],
                use_ttt=False,
            )
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
                cost = 1.0
                loss = float(
                    cross_entropy_loss(
                        out_base["logits"][:, :-1],
                        chunk_batch["input_ids"][:, 1:],
                        chunk_batch["attention_mask"][:, 1:],
                    )
                )

            elif is_rl and isinstance(gating_net, PolicyNetwork):
                if rl_optimizer is None:
                    raise RuntimeError("RL optimizer was not initialized")

                policy_out = gating_net(features, deterministic=True)
                action = int(policy_out["action"][0])
                steps = step_map[action]
                cost = costs_map[action]

                if steps == 0:
                    out_ttt = ttt_model(
                        chunk_batch["input_ids"],
                        attention_mask=chunk_batch["attention_mask"],
                        position_ids=chunk_batch_with_pos["position_ids"],
                        use_ttt=False,
                    )
                    loss = float(
                        cross_entropy_loss(
                            out_ttt["logits"][:, :-1],
                            chunk_batch["input_ids"][:, 1:],
                            chunk_batch["attention_mask"][:, 1:],
                        )
                    )
                else:
                    for _ in range(steps):
                        run_chunk_step(
                            ttt_model,
                            rl_optimizer,
                            chunk_batch_with_pos,
                            use_ttt=True,
                            apply_update=True,
                            ssl_weight=rl_ssl_weight,
                        )
                    eval_metrics = run_chunk_step(
                        ttt_model,
                        None,
                        chunk_batch_with_pos,
                        use_ttt=True,
                        apply_update=False,
                        ssl_weight=rl_ssl_weight,
                    )
                    loss = float(eval_metrics["loss_ce"])

            elif isinstance(gating_net, BinaryGatingNetwork):
                # Binary Gating (Hard Skip) - trained with Gumbel-Softmax
                hard_scale, decision = gating_net.get_decision(features)
                decision_val = int(decision[0])

                if decision_val == 0:
                    # SKIP: use base forward output
                    cost = 1.0
                    loss = float(
                        cross_entropy_loss(
                            out_base["logits"][:, :-1],
                            chunk_batch["input_ids"][:, 1:],
                            chunk_batch["attention_mask"][:, 1:],
                        )
                    )
                else:
                    # UPDATE: run TTT
                    gating_scale = jnp.array([[1.0]], dtype=jnp.float32)  # Full TTT
                    out_ttt = ttt_model(
                        chunk_batch["input_ids"],
                        attention_mask=chunk_batch["attention_mask"],
                        position_ids=chunk_batch_with_pos["position_ids"],
                        use_ttt=True,
                        gating_scale=gating_scale,
                    )
                    cost = 3.0
                    loss = float(
                        cross_entropy_loss(
                            out_ttt["logits"][:, :-1],
                            chunk_batch["input_ids"][:, 1:],
                            chunk_batch["attention_mask"][:, 1:],
                        )
                    )

            elif isinstance(gating_net, GatingNetwork):
                # Continuous Gating with Hard Skip threshold
                scale = float(gating_net(features, train=False)[0, 0])
                scale = max(0.0, scale)

                # Hard Skip: if scale < threshold, skip TTT entirely (cost=1.0)
                if scale < hard_skip_threshold:
                    # Use base forward output (no TTT)
                    cost = 1.0
                    loss = float(
                        cross_entropy_loss(
                            out_base["logits"][:, :-1],
                            chunk_batch["input_ids"][:, 1:],
                            chunk_batch["attention_mask"][:, 1:],
                        )
                    )
                else:
                    # Run TTT with full gating scale
                    gating_scale = jnp.array([[scale]], dtype=jnp.float32)
                    out_ttt = ttt_model(
                        chunk_batch["input_ids"],
                        attention_mask=chunk_batch["attention_mask"],
                        position_ids=chunk_batch_with_pos["position_ids"],
                        use_ttt=True,
                        gating_scale=gating_scale,
                    )
                    # Cost: 1 (base forward already done) + 2 (TTT forward + backward)
                    # When TTT runs, cost is always 3.0 (like UPDATE_1)
                    cost = 3.0
                    loss = float(
                        cross_entropy_loss(
                            out_ttt["logits"][:, :-1],
                            chunk_batch["input_ids"][:, 1:],
                            chunk_batch["attention_mask"][:, 1:],
                        )
                    )
            
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

    # Differentiable Network (optional)
    diff_net: Optional[GatingNetwork] = None
    diff_ttt_model = None
    
    if args.diff_checkpoint:
        diff_net = GatingNetwork(
            config=GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0), 
            rngs=rngs
        )
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
        
        # Load checkpoint without target to avoid strict structure matching
        # This allows loading even if the checkpoint contains extra fields (like optimizer state)
        # or is missing some fields (partial loading)
        ckpt = load_checkpoint(args.diff_checkpoint, target=None)
        
        # Update model state from checkpoint
        # We assume ckpt["state"]["model"] exists and matches trainable_system structure
        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            nnx.update(trainable_system, model_state)
            print("Differentiable Gating and TTT weights loaded.")
        else:
            print("Warning: Could not find 'state.model' in checkpoint. Weights might not be loaded.")
    else:
        print("No Differentiable checkpoint supplied; skipping differentiable evaluation.")

    # RL Network
    rl_net: Optional[PolicyNetwork] = None
    if args.rl_checkpoint:
        rl_net = PolicyNetwork(
            config=PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4),
            rngs=rngs
        )
        print(f"Loading RL Policy checkpoint from {args.rl_checkpoint}...")
        # Load without target
        ckpt = load_checkpoint(args.rl_checkpoint, target=None)
        if "state" in ckpt and "policy" in ckpt["state"]:
            policy_state = unwrap_state(ckpt["state"]["policy"])
            nnx.update(rl_net, policy_state)
            print("RL Policy weights loaded.")
        else:
            print("Warning: Could not find 'state.policy' in checkpoint.")
    else:
        print("No RL checkpoint supplied; skipping RL policy evaluation.")

    # Binary Gating Network (Hard Skip, trained with Gumbel-Softmax)
    binary_net: Optional[BinaryGatingNetwork] = None
    binary_ttt_model = None

    if args.binary_gating_checkpoint:
        binary_net = BinaryGatingNetwork(
            config=BinaryGatingConfig(feature_dim=32, hidden_dim=64, scale_when_update=1.0),
            rngs=rngs
        )
        print(f"Loading Binary Gating (Hard Skip) checkpoint from {args.binary_gating_checkpoint}...")

        # Instantiate fresh TTT model to load weights into
        binary_ttt_model, _ = load_ttt_model(
            model_name=model_name,
            fast_weight_type="ttt",
            seed=args.seed,
            load_pretrained=True,
            vocab_size=vocab_size,
        )

        class TrainableSystemBinary(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None

        trainable_system_binary = TrainableSystemBinary(binary_ttt_model, binary_net)

        ckpt = load_checkpoint(args.binary_gating_checkpoint, target=None)

        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            nnx.update(trainable_system_binary, model_state)
            print("Binary Gating (Hard Skip) and TTT weights loaded.")
        else:
            print("Warning: Could not find 'state.model' in checkpoint.")
    else:
        print("No Binary Gating checkpoint supplied; skipping binary gating evaluation.")

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
    
    # 3. Evaluate Differentiable (with Hard Skip)
    # Use loaded model if available
    if diff_net is not None:
        df_diff = evaluate_model(
            "Differentiable (Hard Skip)",
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
            hard_skip_threshold=args.hard_skip_threshold,
        )
        all_results.append(df_diff)
    
    # 4. Evaluate RL
    # RL uses standard model (policy chooses actions)
    if rl_net is not None:
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
            rl_learning_rate=args.rl_learning_rate,
            rl_max_grad_norm=args.rl_max_grad_norm,
            rl_ssl_weight=args.rl_ssl_weight,
        )
        all_results.append(df_rl)

    # 5. Evaluate Binary Gating (Hard Skip with Gumbel-Softmax)
    if binary_net is not None:
        df_binary = evaluate_model(
            "Binary Gating (Hard Skip)",
            args.model_scale,
            args.budget,
            args.num_eval_batches,
            args.batch_size,
            args.seed,
            binary_net,
            is_rl=False,
            model=binary_ttt_model,
            language=args.language,
            split=args.split,
            num_workers=args.num_workers,
        )
        all_results.append(df_binary)

    # 6. Visualize & Report
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