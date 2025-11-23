"""
Run code generation benchmarks using trained TTT models.

Usage:
    python -m ponderttt.experiments.run_benchmarks --benchmark humaneval --model_scale 125m --checkpoint outputs/my_ckpt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ..data import get_tokenizer
from ..evaluation.benchmarks import BenchmarkSuite, CodeProblem
from ..models import TTTTransformerLM, load_ttt_model
from ..models.gating_nnx import GatingConfig, GatingNetwork
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
            # Forward pass
            # TODO: In real TTT inference, we should process the prompt with TTT update first,
            # then generate with fixed weights (or update continuously).
            # For this smoke test, we run in 'SKIP' mode (Base Model only) or simple forward.
            
            # We use use_ttt=False for stability in this basic generator unless we implement full TTT state management
            outputs = self.model(input_ids, use_ttt=False)
            logits = outputs["logits"][:, -1, :]  # [1, vocab_size]
            
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
        # Simplified loading - assuming we want to load trainable parts
        # Real usage would require the TrainableSystem wrapper if saved that way
        pass # TODO: Implement robust loading based on saved structure
    
    tokenizer = get_tokenizer({"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale])
    
    # Initialize Generator
    generator = SimpleGenerator(model, tokenizer)
    
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
