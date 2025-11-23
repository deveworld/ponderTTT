"""
Run code generation benchmarks using trained TTT models.

Usage:
    python -m ponderttt.experiments.run_benchmarks --benchmark humaneval --model_scale 125m --checkpoint outputs/my_ckpt
"""

import argparse
import json
import os
from pathlib import Path

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
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
            
            pad_len = (mini_batch_size - (current_len % mini_batch_size)) % mini_batch_size
            
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
                out_base = self.model(padded_input, use_ttt=False)
                # Extract features
                # Note: budget_remaining is hard to estimate during generation, using 1.0 (no pressure)
                features = self.feature_extractor.extract(
                    input_ids=padded_input,
                    logits=out_base["logits"],
                    attention_mask=attention_mask,
                    budget_remaining=1.0, 
                )
                
                # Predict scale
                # GatingNetwork vs PolicyNetwork check
                if hasattr(self.gating_net, "evaluate_actions"): # PolicyNetwork
                     policy_out = self.gating_net(features, deterministic=True)
                     action = int(policy_out["action"][0])
                     # Map action to scale (0, 1, 2, 4)
                     step_map = [0, 1, 2, 4]
                     scale = float(step_map[action])
                else: # GatingNetwork
                     scale = float(self.gating_net(features, train=False)[0, 0])
                
                if scale > 0.01:
                    use_ttt = True
                    gating_scale = jnp.array([[scale]])
            
            # Forward pass
            outputs = self.model(padded_input, use_ttt=use_ttt, gating_scale=gating_scale)
            
            # Get logits for the last REAL token
            # padded_input shape [1, L_pad]
            # real token is at index current_len - 1
            logits = outputs["logits"][:, current_len - 1, :]  # [1, vocab_size]
            
            # Sampling
            if temperature > 0:
                probs = jax.nn.softmax(logits / temperature, axis=-1)
                next_token = jax.random.categorical(jax.random.PRNGKey(0), jnp.log(probs)) # seed fixed for simplicity here
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
            # This expects keys: fast_layer, fast_norm, gating_net, etc.
            # And likely wrapped in optimizer state structure.
            # We try to restore into trainable_sys.
            # Note: Since we saved nnx.state(optimizer), the top level keys match the optimizer target.
            # If optimizer wraps TrainableSystem, the keys are the attributes of TrainableSystem.
            
            # We create a temporary target
            target = nnx.state(trainable_sys)
            ckpt = load_checkpoint(args.checkpoint, target=target)
            nnx.update(trainable_sys, ckpt["state"] if "state" in ckpt else ckpt)
            print("Loaded as Differentiable Training checkpoint")
            
            # If we successfully loaded, we might want to use the loaded gating net if not provided separately
            if args.gating_checkpoint is None and "gating_net" in (ckpt["state"] if "state" in ckpt else ckpt):
                 print("Using gating network from model checkpoint")
                 # We need to extract it? No, trainable_sys.gating_net is already updated.
                 # But SimpleGenerator needs it passed explicitly if we want to use it.
                 # However, the generator is init later. We can pass dummy_gating if we want.
                 # But args.gating_checkpoint takes precedence.
                 pass 

        except Exception as e_diff:
            print(f"Not a Differentiable Training checkpoint ({e_diff}), trying Baseline...")
            try:
                # Try loading as Baseline checkpoint (Model structure)
                # Baseline saves nnx.state(optimizer) where optimizer wraps model
                target = nnx.state(model)
                ckpt = load_checkpoint(args.checkpoint, target=target)
                nnx.update(model, ckpt["state"] if "state" in ckpt else ckpt)
                print("Loaded as Baseline checkpoint")
            except Exception as e_base:
                 print(f"Failed to load checkpoint: {e_base}")
                 raise ValueError("Could not load checkpoint as either Differentiable or Baseline format")

    # Load Gating/Policy Checkpoint if provided
    gating_net = None
    if args.gating_checkpoint:
        print(f"Loading gating checkpoint from {args.gating_checkpoint}...")
        # Try PolicyNetwork first
        try:
            from ..models import PolicyConfig, PolicyNetwork
            # Assume default config or infer? hard to infer. using default for now.
            p_config = PolicyConfig(feature_dim=32, hidden_dim=128, num_actions=4)
            p_net = PolicyNetwork(p_config, nnx.Rngs(0))
            target = nnx.state(p_net)
            ckpt = load_checkpoint(args.gating_checkpoint, target=target)
            nnx.update(p_net, ckpt["state"] if "state" in ckpt else ckpt)
            gating_net = p_net
            print("Loaded PolicyNetwork")
        except Exception:
             # Try GatingNetwork
             try:
                 g_config = GatingConfig(feature_dim=32, hidden_dim=64, scale_output=4.0)
                 g_net = GatingNetwork(g_config, nnx.Rngs(0))
                 target = nnx.state(g_net)
                 ckpt = load_checkpoint(args.gating_checkpoint, target=target)
                 nnx.update(g_net, ckpt["state"] if "state" in ckpt else ckpt)
                 gating_net = g_net
                 print("Loaded GatingNetwork")
             except Exception as e:
                 print(f"Failed to load gating checkpoint: {e}")
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
        finally:
            benchmark.problems = original_problems

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
