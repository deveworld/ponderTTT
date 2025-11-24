"""
Run code generation benchmarks using trained TTT models.

Usage:
    python -m ponderttt.experiments.run_benchmarks --benchmark humaneval --model_scale 125m --checkpoint outputs/my_ckpt
"""

import argparse
import json
import os
import traceback
from functools import partial
from pathlib import Path
from typing import Callable, cast

import jax
import jax.numpy as jnp
from flax import nnx

from ponderttt.models.ttt_layer_nnx import TTTConfig

from ..data import get_tokenizer
from ..evaluation.benchmarks import BenchmarkSuite
from ..models import TTTTransformerLM, load_ttt_model
from ..models.gating_nnx import GatingConfig, GatingNetwork
from ..utils import FeatureExtractor
from ..utils.checkpointing import load_checkpoint


# JIT compiled forward pass
# def _generate_step_impl(model, input_ids, use_ttt, gating_scale):
#     return model(input_ids, use_ttt=use_ttt, gating_scale=gating_scale)

# generate_step = cast(Callable, nnx.jit(_generate_step_impl, static_argnames=("use_ttt",)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["humaneval", "mbpp", "classeval", "all"], 
        default="humaneval"
    )
    parser.add_argument("--model_scale", type=str, default="125m")
    parser.add_argument("--checkpoint", type=str, help="Path to trained checkpoint (optional)")
    parser.add_argument("--gating_checkpoint", type=str, help="Path to gating network checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_unsafe", action="store_true", help="Allow unsafe code execution")
    return parser.parse_args()


class SimpleGenerator:
    """Simple autoregressive generator for NNX models."""
    
    def __init__(self, model: TTTTransformerLM, tokenizer, gating_net=None):
        self.model = model
        self.tokenizer = tokenizer
        self.gating_net = gating_net
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        
        # Feature extractor for gating
        self.feature_extractor = FeatureExtractor(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=self.pad_token_id,
            seq_length_norm=512, # approximate norm
        )

        # JIT compiled functions
        @nnx.jit(static_argnames=("use_ttt",))
        def model_forward(model, input_ids, use_ttt, gating_scale):
            return model(input_ids, use_ttt=use_ttt, gating_scale=gating_scale)
        self.model_forward = model_forward

        @nnx.jit(static_argnames=("deterministic",))
        def policy_forward(net, x, deterministic):
            return net(x, deterministic=deterministic)
        self.policy_forward = policy_forward

        @nnx.jit(static_argnames=("train",))
        def gating_forward(net, x, train):
            return net(x, train=train)
        self.gating_forward = gating_forward

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 128, 
        temperature: float = 0.0
    ) -> str:
        """Generate completion for a prompt."""
        # Tokenize
        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = jnp.array([input_ids], dtype=jnp.int32)  # [1, seq_len]
        
        # Generation loop (inefficient, no KV cache)
        for _ in range(max_new_tokens):
            current_len = input_ids.shape[1]
            
            # Prepare TTT inputs (padding if needed)
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
                    remat_group_size = self.model.fast_layer.config.remat_mini_batch_group_size
            
            alignment = mini_batch_size * remat_group_size
            pad_len = (alignment - (current_len % alignment)) % alignment
            
            if pad_len > 0:
                pads = jnp.full((1, pad_len), self.pad_token_id, dtype=jnp.int32)
                padded_input = jnp.concatenate([input_ids, pads], axis=1)
                attention_mask = jnp.concatenate([
                    jnp.ones((1, current_len), dtype=jnp.int32),
                    jnp.zeros((1, pad_len), dtype=jnp.int32)
                ], axis=1)
            else:
                padded_input = input_ids
                attention_mask = jnp.ones((1, current_len), dtype=jnp.int32)
            
            # Determine gating / TTT usage
            use_ttt = False
            gating_scale = None
            
            if self.gating_net is not None:
                # Get baseline output for features
                # We use the JITted forward here too, with use_ttt=False
                out_base = self.model_forward(self.model, padded_input, use_ttt=False, gating_scale=None)
                
                # Extract features
                features = self.feature_extractor.extract(
                    input_ids=padded_input,
                    logits=out_base["logits"],
                    attention_mask=attention_mask,
                    budget_remaining=1.0, 
                )
                
                # Predict scale
                if hasattr(self.gating_net, "evaluate_actions"): # PolicyNetwork
                     policy_out = self.policy_forward(self.gating_net, features, deterministic=True)
                     action = int(policy_out["action"][0])
                     # Map action to scale (0, 1, 2, 4)
                     step_map = [0, 1, 2, 4]
                     scale = float(step_map[action])
                else: # GatingNetwork
                     scale = float(self.gating_forward(self.gating_net, features, train=False)[0, 0])
                
                if scale > 0.01:
                    use_ttt = True
                    gating_scale = jnp.array([[scale]])
            
            # Forward pass
            outputs = self.model_forward(self.model, padded_input, use_ttt=use_ttt, gating_scale=gating_scale)
            
            # Get logits for the last REAL token
            logits = outputs["logits"][:, current_len - 1, :]  # [1, vocab_size]
            
            # Sampling
            if temperature > 0:
                probs = jax.nn.softmax(logits / temperature, axis=-1)
                next_token = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs)) 
            else:
                next_token = jnp.argmax(logits, axis=-1)
            
            next_token = next_token.reshape(1, 1)
            input_ids = jnp.concatenate([input_ids, next_token], axis=1)
            
            if int(next_token[0, 0]) == self.eos_token_id:
                break
                
        # Decode
        full_text = self.tokenizer.decode(input_ids[0])
        # Return only the new part
        completion = full_text[len(prompt):]
        return completion


def unwrap_state(state):
    """Unwrap NNX state dictionary if it contains 'value' keys."""
    if isinstance(state, dict):
        # Check if this dict represents a Variable (has 'value' key)
        # Note: NNX variables serialized by Orbax might appear as {'value': array}
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state


def main():
    args = parse_args()
    
    if args.allow_unsafe:
        os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"
        print("WARNING: Unsafe code execution enabled!")
    
    # Load Model
    print(f"Loading model {args.model_scale}...")
    model, _ = load_ttt_model(
        model_name={"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale],
        fast_weight_type="ttt",
        load_pretrained=True
    )
    
    gating_net = None

    # Load Checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        
        # Helper to reconstruct TrainableSystem for loading diff checkpoints
        class TrainableSystem(nnx.Module):
            def __init__(self, ttt_model, gating_net):
                self.fast_layer = ttt_model.fast_layer
                self.fast_norm = ttt_model.fast_norm
                self.gating_net = gating_net
                if hasattr(ttt_model, 'lm_head'):
                    self.lm_head = ttt_model.lm_head
                else:
                    self.lm_head = None
        
        # Create a dummy gating net to match structure if needed
        dummy_gating = GatingNetwork(GatingConfig(), nnx.Rngs(0))
        trainable_sys = TrainableSystem(model, dummy_gating)
        
        try:
            # Try loading as Differentiable Training checkpoint
            # Use target to ensure correct structure restoration
            print("Attempting to load with strict structure matching...")
            target = {"state": {"model": nnx.state(trainable_sys)}}
            
            ckpt = load_checkpoint(args.checkpoint, target=target)
            
            if "state" in ckpt and "model" in ckpt["state"]:
                nnx.update(trainable_sys, ckpt["state"]["model"])
                print("Loaded as Differentiable Training checkpoint")
            else:
                raise ValueError("Checkpoint does not contain 'state.model'")

            # If we successfully loaded, we might want to use the loaded gating net if not provided separately
            if args.gating_checkpoint is None:
                 print("Using gating network from model checkpoint")
                 gating_net = trainable_sys.gating_net

        except Exception as e_diff:
            print(f"Strict loading failed ({e_diff}), trying loose loading...")
            try:
                # Fallback: Load without target
                ckpt = load_checkpoint(args.checkpoint, target=None)
                
                if "state" in ckpt and "model" in ckpt["state"]:
                    # Try to update trainable_sys with loose dict
                    # Unwrap 'value' keys if present (fix for KeyError: Ellipsis)
                    model_state = unwrap_state(ckpt["state"]["model"])
                    nnx.update(trainable_sys, model_state)
                    print("Loaded as Differentiable Training checkpoint (loose)")
                    if args.gating_checkpoint is None:
                        gating_net = trainable_sys.gating_net
                else:
                    raise ValueError("Checkpoint does not contain 'state.model'")
            except Exception as e_loose:
                print(f"Not a Differentiable Training checkpoint ({e_loose}), trying Baseline...")
                try:
                    # Try loading as Baseline checkpoint (Model structure)
                    # Load without target
                    ckpt = load_checkpoint(args.checkpoint, target=None)
                    
                    if "state" in ckpt and "model" in ckpt["state"]:
                        nnx.update(model, ckpt["state"]["model"])
                        print("Loaded as Baseline checkpoint")
                    else:
                        raise ValueError("Checkpoint does not contain 'state.model'")
                except Exception as e_base:
                     print(f"Failed to load checkpoint: {e_base}")
                     raise ValueError("Could not load checkpoint as either Differentiable or Baseline format")

    # Load Gating/Policy Checkpoint if provided
    if args.gating_checkpoint:
        print(f"Loading gating checkpoint from {args.gating_checkpoint}...")
        # Try PolicyNetwork first
        try:
            from ..models import PolicyConfig, PolicyNetwork
            # Assume default config or infer? hard to infer. using default for now.
            p_config = PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4)
            p_net = PolicyNetwork(p_config, nnx.Rngs(0))
            # Load without target
            ckpt = load_checkpoint(args.gating_checkpoint, target=None)
            if "state" in ckpt and "policy" in ckpt["state"]:
                nnx.update(p_net, ckpt["state"]["policy"])
                gating_net = p_net
                print("Loaded PolicyNetwork")
            else:
                raise ValueError("Checkpoint does not contain 'state.policy'")
        except Exception as e_policy:
             # Try GatingNetwork (part of differentiable training checkpoint)
             try:
                 g_config = GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0)
                 g_net = GatingNetwork(g_config, nnx.Rngs(0))
                 # GatingNetwork is part of the differentiable model, so use model format
                 # Load without target
                 ckpt = load_checkpoint(args.gating_checkpoint, target=None)
                 if "state" in ckpt and "model" in ckpt["state"]:
                     nnx.update(g_net, ckpt["state"]["model"])
                     gating_net = g_net
                     print("Loaded GatingNetwork")
                 else:
                     raise ValueError("Checkpoint does not contain 'state.model'")
             except Exception as e:
                 print(f"Failed to load gating checkpoint as PolicyNetwork ({e_policy}) or GatingNetwork ({e})")
                 raise
    
    tokenizer = get_tokenizer({"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale])

    # Initialize Generator
    generator = SimpleGenerator(model, tokenizer, gating_net)
    
    # Select Benchmarks
    suite = BenchmarkSuite(
        include_humaneval=(args.benchmark in ["humaneval", "all"]),
        include_mbpp=(args.benchmark in ["mbpp", "all"]),
        include_classeval=(args.benchmark in ["classeval", "all"]),
    )
    
    results = {}
    
    # Run Evaluation
    for name, benchmark in suite.benchmarks.items():
        print(f"\nRunning {name} ({len(benchmark)} problems)...")
        
        problems = benchmark.problems
        if args.limit:
            problems = problems[:args.limit]
            print(f"  Limiting to {args.limit} problems")
            
        # Define generation function adapter
        def generate_fn(prompt: str) -> list[str]:
            # Single sample for now
            return [generator.generate(prompt, args.max_new_tokens, args.temperature)]
        
        # Evaluate
        # Note: evaluate() expects the benchmark object to have problems
        # We temporarily patch the benchmark object to respect limit
        original_problems = benchmark.problems
        benchmark.problems = problems
        
        try:
            score = benchmark.evaluate(generate_fn, k=1)
            results[name] = score
            print(f"  Result: {score}")
        except Exception as e:
            print(f"  Failed: {e}")
            traceback.print_exc()
        finally:
            benchmark.problems = original_problems

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
