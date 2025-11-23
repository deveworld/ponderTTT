"""
Train differentiable gating network for adaptive TTT with continuous steps and budget awareness.

Usage:
    python -m ponderttt.experiments.train_differentiable --model_scale 125m --max_steps 4.0 --budget_limit 2.0
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ..data import create_data_iterator, get_tokenizer
from ..models import GPT2Model, load_ttt_model
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor, cross_entropy_loss
from ..utils.checkpointing import save_checkpoint, finalize_checkpointing


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
        default=0.1,
        help="Weight for computational cost penalty (L1 on gating output)",
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

    # Optimizer (No filter needed, as we only included trainable parts)
    optimizer = nnx.Optimizer(
        trainable_system,
        optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.learning_rate),
        ),
        wrt=nnx.All(nnx.Param),
    )
    
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
        seq_length=2048,
        chunk_size=512,
        max_examples=args.batch_size * args.num_iterations * 10,
    )
    
    # Training Step
    @nnx.jit
    def train_step(
        trainable_sys: TrainableSystem,
        optimizer: nnx.Optimizer,
        base_model: GPT2Model, # Passed separately, treated as constant/static if not optimized
        batch: dict, 
        budget_remaining: float,
        beta_ttt: float = 0.1,
    ):
        """
        Differentiable training step with TTT loss.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids
        
        # 1. Base Model Forward (Frozen)
        # We stop gradient to ensure no grads flow to base_model
        hidden_states = jax.lax.stop_gradient(base_model(input_ids, train=False))
        
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
                position_ids=None,
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
    for i, sequence_batch in enumerate(data_iter):
        if i >= args.num_iterations:
            break
            
        # Process chunks in sequence
        chunks = sequence_batch["chunks"] # [B, num_chunks, chunk_size]
        masks = sequence_batch["chunk_attention_mask"]
        
        num_chunks = chunks.shape[1]
        
        # Per-sequence budget tracking
        # budget_limit is avg steps per chunk. Total budget = budget_limit * num_chunks
        total_budget = args.budget_limit * num_chunks
        current_spend = 0.0
        
        avg_gate_val = 0.0
        avg_ce_loss = 0.0
        
        # Iterate through chunks in the sequence
        # Note: In a real "Budget-Aware" setting, we should carry over TTT state if chunks are dependent.
        # But here we treat chunks as segments of a stream. 
        # Since TTTLayer resets per call (stateless usage in this script), we treat them as independent
        # except for the budget state.
        
        # Update: TTTLayer logic IS stateless w.r.t weights unless we pass state explicitly or use stateful mode.
        # `load_ttt_model` gives us a stateful NNX model.
        # However, `train_step` calls `ttt_model(...)`.
        # If we want proper TTT history (fast weights accumulating), we should NOT reset it.
        # But `train_step` might need to handle state carefully with gradients.
        # For now, let's assume chunks are processed independently (standard TTT-chunk approach) 
        # or effectively "resetting" fast weights each chunk for simplicity of gradient implementation.
        # (Full BPTT through multiple chunks is heavy).
        
        feature_extractor.reset_history()
        
        # Initialize stats for logging
        loss, l_ttt, cost = 0.0, 0.0, 0.0
        avg_ce_loss, avg_gate_val = 0.0, 0.0
        
        for c_idx in range(num_chunks):
            chunk_batch = {
                "input_ids": chunks[:, c_idx],
                "attention_mask": masks[:, c_idx]
            }
            
            # Calculate remaining budget fraction
            if total_budget > 1e-6:
                budget_rem_fraction = max(0.0, (total_budget - current_spend) / total_budget)
            else:
                budget_rem_fraction = 0.0
                
            loss, ce, l_ttt, cost, gate = train_step(
                trainable_system, 
                optimizer, 
                ttt_model.base_model,
                chunk_batch, 
                budget_rem_fraction
            )
            
            # Accumulate spend (using the mean gate value of the batch as proxy for budget tracking)
            # Ideally, we track per-sample, but we use batch mean for simple features.
            current_spend += float(gate)
            
            avg_gate_val += float(gate)
            avg_ce_loss += float(ce)
            
            # Update feature extractor history
            feature_extractor.update_history(float(ce), float(gate))
            
        avg_gate_val /= num_chunks
        avg_ce_loss /= num_chunks
        
        print(f"Iter {i+1}: Loss={loss:.4f}, CE={avg_ce_loss:.4f}, L_TTT={float(l_ttt):.4f}, Gate={avg_gate_val:.4f}, RemBudget={1.0 - (current_spend/total_budget):.2f}")
        
        history.append({
            "iteration": i,
            "loss": float(loss),
            "ce_loss": float(avg_ce_loss),
            "l_ttt": float(l_ttt),
            "gate_mean": float(avg_gate_val),
            "budget_utilization": current_spend / total_budget
        })

    # Save results
    with open(output_dir / "history_continuous.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save Checkpoint
    print("Saving checkpoint...")
    train_state = nnx.state(trainable_system)
    save_checkpoint(
        checkpoint_dir=output_dir,
        step=args.num_iterations,
        state=train_state,
        metadata={
            "model_scale": args.model_scale,
            "max_steps": args.max_steps,
            "budget_limit": args.budget_limit,
        }
    )
    finalize_checkpointing()
    print(f"Checkpoint saved to {output_dir}")

if __name__ == "__main__":
    main()
