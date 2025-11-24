"""
Train differentiable gating network for adaptive TTT with continuous steps and budget awareness.

Usage:
    python -m ponderttt.experiments.train_differentiable --model_scale 125m --max_steps 4.0 --budget_limit 2.0
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ..data import create_data_iterator, get_tokenizer
from ..models import GPT2Model, load_ttt_model
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import save_checkpoint, load_checkpoint, finalize_checkpointing
import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train differentiable gating (NNX)")

    parser.add_argument(
        "--model_scale",
        type=str,
        choices=["125m", "350m", "1b"],
        default="125m",
        help="Model scale",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations (batches)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for gating network",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (sequences)",
    )
    parser.add_argument(
        "--cost_weight",
        type=float,
        default=0.01,
        help="Weight for computational cost penalty (L1 on gating output)",
    )
    parser.add_argument(
        "--reward_shaping",
        action="store_true",
        default=True,
        help="Use reward shaping: penalize cost only when no performance gain",
    )
    parser.add_argument(
        "--max_steps",
        type=float,
        default=4.0,
        help="Maximum scaling factor for TTT update (approx. max steps)",
    )
    parser.add_argument(
        "--budget_limit",
        type=float,
        default=2.0,
        help="Target average steps per chunk (used for budget feature calculation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/differentiable",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
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
        default=100,
        help="Save checkpoint every N steps",
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
    
    return parser.parse_args()


def get_model_name(model_scale: str) -> str:
    mapping = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
    }
    return mapping[model_scale]


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PonderTTT Differentiable Gating Training (Continuous & Budget-Aware)")
    print("=" * 60)
    print(f"Model scale: {args.model_scale}")
    print(f"Max steps: {args.max_steps}")
    print(f"Budget limit (avg steps): {args.budget_limit}")
    print(f"Cost weight (gamma): {args.cost_weight}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = get_model_name(args.model_scale)
    tokenizer = get_tokenizer(model_name)
    
    # Load TTT Model (Frozen Backbone)
    print("Loading TTT model...")
    ttt_model, ttt_config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        seed=args.seed,
        load_pretrained=True,
        vocab_size=tokenizer.get_vocab_size(),
    )
    
    # Initialize Gating Network
    print("Initializing Gating Network...")
    gating_config = GatingConfig(
        feature_dim=32,
        hidden_dim=64,
        scale_output=args.max_steps, 
    )
    rngs = nnx.Rngs(args.seed + 1)
    gating_net = GatingNetwork(config=gating_config, rngs=rngs)
    
    # System container for optimization (Only Trainable Parts)
    class TrainableSystem(nnx.Module):
        def __init__(self, ttt_model, gating_net):
            # Share references to trainable parts
            self.fast_layer = ttt_model.fast_layer
            self.fast_norm = ttt_model.fast_norm
            self.gating_net = gating_net
            # Handle LM Head
            if hasattr(ttt_model, 'lm_head'):
                self.lm_head = ttt_model.lm_head
            else:
                self.lm_head = None

    trainable_system = TrainableSystem(ttt_model, gating_net)

    # Initialize WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"diff_{args.model_scale}_budget{args.budget_limit}",
        )

    # Optimizer (No filter needed, as we only included trainable parts)
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.learning_rate),
        ),
        wrt=nnx.All(nnx.Param),
    )
    
    # Resume from checkpoint if requested
    start_iteration = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        # Create target structure for correct loading
        target = {
            "state": {"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
            "step": 0,
            "metadata": {
                "model_scale": "",
                "max_steps": 0.0,
                "budget_limit": 0.0,
            }
        }
        ckpt = load_checkpoint(args.resume_from, target=target)
        nnx.update(trainable_system, ckpt["state"]["model"])
        nnx.update(optimizer, ckpt["state"]["optimizer"])
        start_iteration = ckpt.get("step", 0)
        print(f"Resumed from iteration {start_iteration}")

    
    # Feature Extractor
    feature_extractor = FeatureExtractor(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )
    
    # Data Iterator
    data_iter = create_data_iterator(
        tokenizer=tokenizer,
        split="train",
        batch_size=args.batch_size,
        seq_length=1024,
        chunk_size=512,
        max_examples=args.batch_size * args.num_iterations * 2,
        num_workers=args.num_workers,
    )
    
    # Training Step
    @nnx.jit
    def train_step(
        trainable_sys: TrainableSystem,
        optimizer: nnx.Optimizer,
        base_model: GPT2Model, # Passed separately, treated as constant/static if not optimized
        batch: dict, 
        budget_remaining: jax.Array,
        baseline_loss: jax.Array,  # Previous chunk's loss for reward shaping
        beta_ttt: float = 0.1,
    ):
        """
        Differentiable training step with TTT loss and reward shaping.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch.get("position_ids")
        labels = input_ids
        
        # 1. Base Model Forward (Frozen)
        # We stop gradient to ensure no grads flow to base_model
        hidden_states = jax.lax.stop_gradient(base_model(
            input_ids, 
            position_ids=position_ids,
            train=False
        ))
        
        # Compute logits for features (also frozen base)
        if ttt_model.tie_word_embeddings:
            embedding_kernel = base_model.wte.embedding[...]  # type: ignore
            logits_base = hidden_states @ embedding_kernel.T
        else:
            # If lm_head is separate, it might be in trainable_sys or not?
            # For GPT-2, usually tied. If untied, lm_head is in trainable_sys?
            # Let's assume tied for now or use base logic.
            # If trainable_sys has lm_head, use it.
            if trainable_sys.lm_head:
                logits_base = trainable_sys.lm_head(hidden_states)
            else:
                 # Fallback if not in sys (e.g. untied but frozen? Unlikely)
                 logits_base = hidden_states # Placeholder
        
        features = feature_extractor.extract(
            input_ids=input_ids,
            logits=logits_base,
            hidden_states=[hidden_states],
            attention_mask=attention_mask,
            budget_remaining=budget_remaining,
        )
        features = jax.lax.stop_gradient(features)
        
        def loss_fn(sys):
            # 2. Predict Gating
            # Assuming gating_net is callable via __call__
            gating_scale = sys.gating_net(features, train=True)
            
            # 3. Apply TTT (Fast Path)
            # Replicate TTTTransformerLM logic with sys modules
            hidden_states_normed = sys.fast_norm(hidden_states)
            
            fast_output, fast_stats = sys.fast_layer(
                hidden_states_normed,
                mask=attention_mask,
                position_ids=position_ids,
                train=True,
                gating_scale=gating_scale
            )
            
            adapted_hidden = hidden_states + fast_output
            
            # Final Logits
            if ttt_model.tie_word_embeddings:
                embedding_kernel = base_model.wte.embedding[...]  # type: ignore
                logits = adapted_hidden @ embedding_kernel.T
            elif sys.lm_head:
                logits = sys.lm_head(adapted_hidden)
            else:
                raise ValueError("LM Head not found")

            # 4. Compute Losses
            ce_loss = cross_entropy_loss(logits[:, :-1], labels[:, 1:], attention_mask[:, 1:])
            
            if fast_stats is not None and "ttt_loss_step_1" in fast_stats:
                l_ttt = jnp.mean(fast_stats["ttt_loss_step_1"])
                if l_ttt is None:
                    l_ttt = 0.0
            else:
                l_ttt = 0.0
            
            # Reward Shaping: Penalize cost only when no improvement
            # If ce_loss < baseline_loss: reward (negative penalty)
            # If ce_loss >= baseline_loss: penalty proportional to cost and lack of improvement
            if args.reward_shaping:
                improvement = jnp.maximum(0.0, baseline_loss - ce_loss)  # Positive if better
                waste = jnp.maximum(0.0, ce_loss - baseline_loss)  # Positive if worse
                
                # Cost efficiency: penalize high cost with low/negative improvement
                # Reward: encourage using compute when it helps
                cost_term = jnp.mean(gating_scale)
                
                # Dynamic penalty based on budget urgency
                # If budget is full (remaining=1.0), urgency=0.0 -> low base penalty
                # If budget is empty (remaining=0.0), urgency=1.0 -> high base penalty
                # If budget is negative (remaining<0.0), urgency>1.0 -> very high penalty
                budget_urgency = 1.0 - budget_remaining
                
                # Simple polynomial scaling for cost factor
                # When urgency=0 (full budget), factor = 0.05
                # When urgency=1 (empty budget), factor = 0.05 + 5.0 = 5.05
                # When urgency=1.5 (overdraft), factor = 0.05 + 5.0 * 3.375 = 16.9
                base_cost_factor = 0.05 + 5.0 * (jnp.maximum(0.0, budget_urgency) ** 3)
                
                efficiency_penalty = cost_term * (base_cost_factor + waste * 5.0)
                efficiency_reward = improvement * cost_term * 2.0
                
                cost_penalty = (efficiency_penalty - efficiency_reward) * args.cost_weight
            else:
                cost_penalty = jnp.mean(gating_scale) * args.cost_weight
            
            total_loss = ce_loss + (beta_ttt * l_ttt) + cost_penalty
            
            return total_loss, (ce_loss, l_ttt, cost_penalty, jnp.mean(gating_scale))

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (ce_loss, l_ttt, cost_loss, avg_gate)), grads = grad_fn(trainable_sys)
        
        optimizer.update(trainable_sys, grads)
        
        return loss, ce_loss, l_ttt, cost_loss, avg_gate

    # History tracking
    history = []
    
    print("Starting training...")
    # Skip data iterator to resume point if needed
    if start_iteration > 0:
        print(f"Skipping first {start_iteration} batches to resume training...")
    
    iter_count = start_iteration
    for i, sequence_batch in enumerate(data_iter):
        # Skip batches that were already processed
        if i < start_iteration:
            continue

        if iter_count >= args.num_iterations:
            break
            
        # Process chunks in sequence
        chunks = sequence_batch["chunks"] # [B, num_chunks, chunk_size]
        masks = sequence_batch["chunk_attention_mask"]
        
        num_chunks = chunks.shape[1]
        
        # Per-sequence budget tracking
        total_budget = args.budget_limit * num_chunks
        current_spend = 0.0
        
        avg_gate_val = 0.0
        avg_ce_loss = 0.0
        
        feature_extractor.reset_history()
        
        # Initialize stats for logging
        loss, l_ttt, cost = 0.0, 0.0, 0.0
        avg_ce_loss, avg_gate_val = 0.0, 0.0
        prev_ce_loss = jnp.array(3.0)  # Initialize with reasonable baseline
        
        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx],
                "position_ids": jnp.arange(
                    c_idx * 512, # hardcoded chunk_size, consider using args/config
                    (c_idx + 1) * 512,
                    dtype=jnp.int32
                )[None, :].repeat(chunks.shape[0], axis=0)
            }
            
            # Check for valid tokens
            num_valid_tokens = jnp.sum(chunk_batch["attention_mask"][:, 1:])
            if num_valid_tokens < 16:
                continue
            
            # Calculate remaining budget fraction
            if total_budget > 1e-6:
                # Allow negative values to signal budget violation
                budget_rem_fraction = (total_budget - current_spend) / total_budget
            else:
                budget_rem_fraction = 0.0
                
            loss, ce, l_ttt, cost, gate = train_step(
                trainable_system, 
                optimizer, 
                ttt_model.base_model,
                chunk_batch, 
                jnp.array(budget_rem_fraction),
                prev_ce_loss,
            )
            
            # Stability check
            if not jnp.isfinite(loss) or float(ce) > 20.0:
                print(f"Warning: Skipping unstable batch (loss={loss:.4f}, ce={ce:.4f})")
                continue
            
            # Update baseline for next chunk
            prev_ce_loss = ce
            
            current_spend += float(gate)
            avg_gate_val += float(gate)
            avg_ce_loss += float(ce)
            
            feature_extractor.update_history(float(ce), float(gate))
            
        avg_gate_val /= max(1, num_chunks) # prevent div by zero if all skipped, though unlikely
        avg_ce_loss /= max(1, num_chunks)
        perplexity = math.exp(avg_ce_loss)
        budget_util = current_spend / max(1e-6, total_budget)
        
        print(f"Iter {iter_count+1}: Loss={loss:.4f}, CE={avg_ce_loss:.4f}, PPL={perplexity:.2f}, L_TTT={float(l_ttt):.4f}, Gate={avg_gate_val:.4f}, RemBudget={1.0 - budget_util:.2f}")
        
        # WandB Logging
        if args.wandb_project:
            wandb.log({
                f"seed_{args.seed}/loss": float(loss),
                f"seed_{args.seed}/ce_loss": float(avg_ce_loss),
                f"seed_{args.seed}/perplexity": perplexity,
                f"seed_{args.seed}/l_ttt": float(l_ttt),
                f"seed_{args.seed}/gate_mean": float(avg_gate_val),
                f"seed_{args.seed}/budget_utilization": budget_util,
                "iteration": iter_count + 1,
            })

        history.append({
            "iteration": iter_count,
            "loss": float(loss),
            "ce_loss": float(avg_ce_loss),
            "perplexity": perplexity,
            "l_ttt": float(l_ttt),
            "gate_mean": float(avg_gate_val),
            "budget_utilization": budget_util
        })
        
        # Periodic Checkpoint
        if (iter_count + 1) % args.save_every == 0 and (iter_count + 1) < args.num_iterations:
            print(f"Saving checkpoint at iter {iter_count + 1}...")
            save_checkpoint(
                checkpoint_dir=output_dir,
                step=iter_count + 1,
                state={"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
                metadata={
                    "model_scale": args.model_scale,
                    "max_steps": args.max_steps,
                    "budget_limit": args.budget_limit,
                }
            )
            # Also save to WandB if enabled
            # if args.wandb_project:
            #     wandb.save(str(output_dir / f"checkpoint_{iter_count + 1}/*"))

        iter_count += 1

    # Save results
    with open(output_dir / "history_continuous.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save Final Checkpoint
    print("Saving final checkpoint...")
    save_checkpoint(
        checkpoint_dir=output_dir,
        step=iter_count,
        state={"model": nnx.state(trainable_system), "optimizer": nnx.state(optimizer)},
        metadata={
            "model_scale": args.model_scale,
            "max_steps": args.max_steps,
            "budget_limit": args.budget_limit,
        }
    )
    finalize_checkpointing()
    print(f"Checkpoint saved to {output_dir}")
    
    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()
