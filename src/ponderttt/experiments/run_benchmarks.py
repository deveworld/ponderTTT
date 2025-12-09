"""
Run code generation benchmarks using trained TTT models.

Usage:
    python -m ponderttt.experiments.run_benchmarks --benchmark humaneval --model_scale 125m --checkpoint outputs/my_ckpt
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ..data import get_tokenizer
from ..evaluation.benchmarks import BenchmarkSuite, _check_solution
from ..models import TTTModel, load_ttt_model, TTTConfig
from ..utils.checkpointing import load_checkpoint, unwrap_state


@nnx.jit(static_argnames=("use_ttt",))
def _model_forward_jit(model, input_ids, use_ttt):
    return model(input_ids, use_ttt=use_ttt)


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["humaneval", "mbpp", "all"],
        default="humaneval"
    )
    parser.add_argument("--model_scale", type=str, default="125m")
    parser.add_argument("--checkpoint", type=str, help="Path to trained TTT checkpoint (optional)")
    parser.add_argument("--baseline", type=str, help="Path to baseline model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_unsafe", action="store_true", help="Allow unsafe code execution")
    parser.add_argument("--use_ttt", action="store_true", help="Enable TTT (Test-Time Training) updates during generation")
    return parser.parse_args()


class SimpleGenerator:
    """Simple autoregressive generator for NNX models."""

    def __init__(
        self,
        model: TTTModel,
        tokenizer,
        use_ttt: bool = False,
        rng_key: jax.Array | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.use_ttt = use_ttt
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # GPT-2 variants cap usable context via learned position embeddings
        base_config = getattr(model, "gpt2_config", None)
        if base_config is None and hasattr(model, "base_model"):
            base_config = getattr(model.base_model, "config", None)
        self.max_seq_len = getattr(base_config, "n_positions", 1024)
        self._warned_truncation = False

    def _next_rng_key(self) -> jax.Array:
        """Split and return the next PRNG key for sampling."""
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def _get_target_length(self, current_len: int, alignment: int) -> int:
        """Calculate target length using power-of-2 buckets to minimize recompilation."""
        # Buckets: 64, 128, 256, 512, 1024, ...
        target = 64
        while target < current_len:
            target *= 2

        # Ensure alignment
        if target % alignment != 0:
            target = ((target + alignment - 1) // alignment) * alignment

        return min(target, self.max_seq_len)

    def _call_model(self, input_tensor: jax.Array):
        """Run model forward pass."""
        return _model_forward_jit(self.model, input_tensor, use_ttt=self.use_ttt)

    def _truncate_context(self, token_ids: list[int]) -> list[int]:
        """Clamp context to the model's positional window."""
        if len(token_ids) <= self.max_seq_len:
            return token_ids
        if not self._warned_truncation:
            print(
                f"WARNING: Prompt length {len(token_ids)} exceeds model limit {self.max_seq_len}. "
                "Keeping the most recent tokens."
            )
            self._warned_truncation = True
        return token_ids[-self.max_seq_len:]

    def _clean_completion(self, completion: str, prompt: str | None = None) -> str:
        """
        Clean up the generated completion by truncating at stop sequences.
        This prevents syntax errors caused by the model rambling on.
        """
        # Common stop sequences for code generation
        stop_sequences = ["\ndef ", "\nclass ", "\n#", "\nif __name__", "\n\n\n"]

        # Find the earliest occurrence of any stop sequence
        min_idx = len(completion)

        for seq in stop_sequences:
            idx = completion.find(seq)
            if idx != -1 and idx < min_idx:
                min_idx = idx

        if min_idx < len(completion):
            completion = completion[:min_idx]

        return completion

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        """Generate completion for a prompt."""
        # Tokenize
        prompt_token_ids = self._truncate_context(self.tokenizer.encode(prompt).ids)
        input_ids = jnp.array([prompt_token_ids], dtype=jnp.int32)  # [1, seq_len]
        generated_tokens: list[int] = []

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

            # Use bucketed padding to reduce recompilation
            target_len = self._get_target_length(current_len, alignment)
            pad_len = target_len - current_len

            if pad_len > 0:
                pads = jnp.full((1, pad_len), self.pad_token_id, dtype=jnp.int32)
                padded_input = jnp.concatenate([input_ids, pads], axis=1)
            else:
                padded_input = input_ids

            outputs = self._call_model(padded_input)

            # Get logits for the last REAL token
            logits = outputs["logits"][:, current_len - 1, :]  # [1, vocab_size]

            # Sampling
            if temperature > 0:
                key = self._next_rng_key()
                scaled_logits = logits / float(temperature)
                next_token = jax.random.categorical(key, scaled_logits, axis=-1)
            else:
                next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token.reshape(1, 1)
            token_val = int(next_token[0, 0])
            if token_val != self.eos_token_id:
                generated_tokens.append(token_val)
            input_ids = jnp.concatenate([input_ids, next_token], axis=1)
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            if token_val == self.eos_token_id:
                break

        # Decode
        completion = self.tokenizer.decode(generated_tokens)
        return self._clean_completion(completion, prompt)

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        # Tokenize
        input_ids_list = [self._truncate_context(list(self.tokenizer.encode(p).ids)) for p in prompts]
        batch_size = len(prompts)
        generated_tokens = [[] for _ in range(batch_size)]

        # Track which sequences are finished
        finished = [False] * batch_size

        for _ in range(max_new_tokens):
            if all(finished):
                break

            # Prepare batch
            for idx in range(batch_size):
                input_ids_list[idx] = self._truncate_context(input_ids_list[idx])

            current_lens = [len(ids) for ids in input_ids_list]
            max_len = max(current_lens)

            # Alignment
            mini_batch_size = 16
            remat_group_size = 1
            if hasattr(self.model, "fast_layer") and hasattr(self.model.fast_layer, "config"):
                if isinstance(self.model.fast_layer.config, TTTConfig):
                    mini_batch_size = self.model.fast_layer.config.mini_batch_size
                    remat_group_size = self.model.fast_layer.config.remat_mini_batch_group_size

            alignment = mini_batch_size * remat_group_size

            # Use bucketed padding to reduce recompilation
            target_len = self._get_target_length(max_len, alignment)

            # Construct tensors
            padded_input_ids = []

            for i, ids in enumerate(input_ids_list):
                pad_len = target_len - len(ids)
                padded_ids = ids + [self.pad_token_id] * pad_len
                padded_input_ids.append(padded_ids)

            input_tensor = jnp.array(padded_input_ids, dtype=jnp.int32)

            outputs = self._call_model(input_tensor)

            # Extract logits
            indices = jnp.array([length - 1 for length in current_lens])  # [B]
            next_token_logits = outputs["logits"][jnp.arange(batch_size), indices, :]  # [B, V]

            # Sampling
            if temperature > 0:
                key = self._next_rng_key()
                scaled_logits = next_token_logits / float(temperature)
                next_tokens = jax.random.categorical(key, scaled_logits, axis=-1)
            else:
                next_tokens = jnp.argmax(next_token_logits, axis=-1)

            # Update sequences
            for i, token in enumerate(next_tokens):
                if not finished[i]:
                    token_val = int(token)
                    if token_val != self.eos_token_id:
                        generated_tokens[i].append(token_val)
                    if token_val == self.eos_token_id:
                        finished[i] = True
                    else:
                        input_ids_list[i].append(token_val)
                        if len(input_ids_list[i]) > self.max_seq_len:
                            # Keep only the most recent tokens within context window
                            input_ids_list[i] = input_ids_list[i][-self.max_seq_len:]

        # Decode
        completions = []
        for i in range(batch_size):
            completion = self.tokenizer.decode(generated_tokens[i])
            completions.append(self._clean_completion(completion, prompts[i]))

        return completions


def main():
    args = parse_args()

    if args.checkpoint and args.baseline:
        raise ValueError("Please provide only one of --checkpoint or --baseline")

    if args.allow_unsafe:
        os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"
        print("WARNING: Unsafe code execution enabled!")
    elif os.environ.get("PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS") != "1":
        print("\nERROR: Benchmarks require executing generated code.")
        print("Please use --allow_unsafe to enable code execution.")
        print("Example: python -m ponderttt.experiments.run_benchmarks --allow_unsafe ...\n")
        return

    tokenizer = get_tokenizer({"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale])
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    vocab_size = tokenizer.get_vocab_size()

    # Load Model
    print(f"Loading model {args.model_scale}...")
    model, _ = load_ttt_model(
        model_name={"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[args.model_scale],
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id
    )

    # Load Checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = load_checkpoint(args.checkpoint, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            # Only update fast_layer and fast_norm (TTT components)
            if "fast_layer" in model_state:
                nnx.update(model.fast_layer, model_state["fast_layer"])
                print("  ✓ Loaded fast_layer from checkpoint")
            if "fast_norm" in model_state:
                nnx.update(model.fast_norm, model_state["fast_norm"])
                print("  ✓ Loaded fast_norm from checkpoint")
        else:
            raise ValueError("Checkpoint does not contain 'state.model'")
    elif args.baseline:
        print(f"Loading baseline checkpoint from {args.baseline}...")
        ckpt = load_checkpoint(args.baseline, target=None)
        if "state" in ckpt and "model" in ckpt["state"]:
            model_state = unwrap_state(ckpt["state"]["model"])
            # Only update fast_layer and fast_norm (TTT components)
            if "fast_layer" in model_state:
                nnx.update(model.fast_layer, model_state["fast_layer"])
                print("  ✓ Loaded fast_layer from checkpoint")
            if "fast_norm" in model_state:
                nnx.update(model.fast_norm, model_state["fast_norm"])
                print("  ✓ Loaded fast_norm from checkpoint")
        else:
            raise ValueError("Baseline checkpoint does not contain 'state.model'")

    # Initialize Generator
    print(f"TTT enabled: {args.use_ttt}")
    generator = SimpleGenerator(
        model,
        tokenizer,
        use_ttt=args.use_ttt,
        rng_key=jax.random.PRNGKey(args.seed),
    )
    
    # Select Benchmarks
    suite = BenchmarkSuite(
        include_humaneval=(args.benchmark in ["humaneval", "all"]),
        include_mbpp=(args.benchmark in ["mbpp", "all"]),
    )
    
    results = {}
    
    # Run Evaluation
    for name, benchmark in suite.benchmarks.items():
        print(f"\nRunning {name} ({len(benchmark)} problems)...")
        
        problems = benchmark.problems
        if args.limit:
            problems = problems[:args.limit]
            print(f"  Limiting to {args.limit} problems")
            
        print(f"  Processing in batches of {args.batch_size}...")
        
        scores = []
        attempts_per_problem = []
        
        # Batch processing loop
        for i in tqdm(range(0, len(problems), args.batch_size), desc=f"Running {name}"):
            batch_problems = problems[i : i + args.batch_size]
            prompts = [p.prompt for p in batch_problems]
            
            try:
                completions = generator.generate_batch(prompts, args.max_new_tokens, args.temperature)
                
                for problem, completion in zip(batch_problems, completions):
                    # Check correctness (k=1)
                    is_correct = _check_solution(problem, completion)
                    scores.append(1.0 if is_correct else 0.0)
                    attempts_per_problem.append(1)
                    
            except Exception as e:
                print(f"  Batch failed: {e}")
                traceback.print_exc()
                # Fill with failures
                for _ in batch_problems:
                    scores.append(0.0)
                    attempts_per_problem.append(0)
            
            # Progress update
            # current_avg = sum(scores) / len(scores) if scores else 0.0
            # print(f"  Batch {i//args.batch_size + 1}/{(len(problems)+args.batch_size-1)//args.batch_size} done. Current Pass@1: {current_avg:.2%}")

        # Final results
        final_score = sum(scores) / len(scores) if scores else 0.0
        avg_attempts = sum(attempts_per_problem) / len(attempts_per_problem) if attempts_per_problem else 0.0
        results[name] = {
            "pass@1": final_score,
            "num_problems": len(scores),
            "avg_attempts": avg_attempts,
        }
        print(f"  Result: Pass@1 = {final_score:.4f} ({len(scores)} problems)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/results.json")

if __name__ == "__main__":
    main()
