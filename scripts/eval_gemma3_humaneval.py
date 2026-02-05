#!/usr/bin/env python3
"""
Evaluate Gemma 3 on HumanEval with and without TTT.

Uses fixed-size chunks for JIT efficiency (same pattern as compare_methods.py).

Usage:
    python scripts/eval_gemma3_humaneval.py --model_scale 4b --use_ttt
    python scripts/eval_gemma3_humaneval.py --model_scale 4b --no_ttt
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ["PONDER_TTT_ALLOW_UNSAFE_BENCHMARKS"] = "1"
os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from ponderttt.data import get_tokenizer
from ponderttt.evaluation.benchmarks import HumanEvalBenchmark
from ponderttt.models import load_ttt_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_model_name(scale: str) -> str:
    """Get model name from scale."""
    return f"gemma3-{scale}"


def get_default_tokenizer(model_scale: str) -> str:
    """Get default tokenizer for model scale."""
    return f"google/gemma-3-{model_scale}-it"


def format_prompt_for_it_model(prompt: str) -> str:
    """Format HumanEval prompt for instruction-tuned model.

    Gemma 3 IT expects: <start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model
    """
    instruction = f"Complete the following Python function. Only output the function body, no explanations.\n\n```python\n{prompt}```"
    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n```python\n{prompt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gemma 3 on HumanEval")
    parser.add_argument(
        "--model_scale",
        choices=["1b", "4b", "12b", "27b"],
        default="4b",
        help="Model scale",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to trained checkpoint (for TTT models)",
    )
    parser.add_argument(
        "--use_ttt",
        action="store_true",
        help="Enable TTT during inference",
    )
    parser.add_argument(
        "--no_ttt",
        action="store_true",
        help="Disable TTT (baseline)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/humaneval",
        help="Output directory",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to evaluate (None = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Cache size for KV cache (prompt + max_new_tokens)",
    )
    return parser.parse_args()


def make_forward_fns(use_ttt: bool):
    """Create separate forward functions for prefill and decode.

    TTT requires sequence length divisible by mini_batch_size (16).
    For single-token decode, we must disable TTT.

    Returns two functions (to be JIT compiled separately):
    - prefill_fn: Uses TTT if enabled (for full prompt processing)
    - decode_fn: Always disables TTT (for single-token generation)
    """

    def prefill_fn(
        model: Any,
        input_ids: jax.Array,
        position_ids: jax.Array,
        cache: dict | None = None,
    ) -> tuple[jax.Array, dict | None]:
        """Forward pass for prefill with TTT enabled."""
        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
            use_ttt=use_ttt,  # Use TTT for prefill
        )
        if isinstance(result, tuple):
            outputs, new_cache = result
        else:
            outputs, new_cache = result, None
        return outputs["logits"], new_cache

    def decode_fn(
        model: Any,
        input_ids: jax.Array,
        position_ids: jax.Array,
        cache: dict | None = None,
    ) -> tuple[jax.Array, dict | None]:
        """Forward pass for decode without TTT (single tokens)."""
        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=None,
            cache=cache,
            use_ttt=False,  # Always disable TTT for single-token decode
        )
        if isinstance(result, tuple):
            outputs, new_cache = result
        else:
            outputs, new_cache = result, None
        return outputs["logits"], new_cache

    return prefill_fn, decode_fn


def create_generate_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    use_ttt: bool,
    num_samples: int,
    chunk_size: int,
):
    """Create generation function for HumanEval with KV cache for efficiency."""

    # Get EOS token ID
    eos_id = (
        tokenizer.token_to_id("<eos>")
        or tokenizer.token_to_id("</s>")
        or tokenizer.token_to_id("<end_of_turn>")
        or 1
    )

    # Create and JIT compile separate forward functions for prefill and decode
    prefill_fn, decode_fn = make_forward_fns(use_ttt)
    jit_prefill = nnx.jit(prefill_fn)
    jit_decode = nnx.jit(decode_fn)

    # Calculate cache size (prompt + max_new_tokens)
    cache_size = chunk_size

    # Warmup JIT with prefill (full prompt) and decode (single token) shapes
    logger.info(f"Warming up JIT with cache_size={cache_size}...")

    # Warmup prefill (length must be multiple of 16 for TTT)
    dummy_prefill_len = (chunk_size // 2 // 16) * 16  # Round down to multiple of 16
    if dummy_prefill_len == 0:
        dummy_prefill_len = 16
    dummy_prefill_ids = jnp.ones((1, dummy_prefill_len), dtype=jnp.int32)
    dummy_prefill_pos = jnp.arange(dummy_prefill_len, dtype=jnp.int32)[None, :]
    cache = model.init_cache(cache_size=cache_size, batch_size=1)
    _, cache = jit_prefill(model, dummy_prefill_ids, dummy_prefill_pos, cache)

    # Warmup decode (single token)
    dummy_decode_ids = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_decode_pos = jnp.array([[dummy_prefill_len]], dtype=jnp.int32)
    _ = jit_decode(model, dummy_decode_ids, dummy_decode_pos, cache)

    logger.info("JIT warmup complete (prefill + decode).")

    def generate(prompt: str) -> list[str]:
        """Generate completions using KV cache for efficiency."""
        # Apply chat template for IT models
        formatted_prompt = format_prompt_for_it_model(prompt)

        # Tokenize the formatted prompt
        encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        prompt_ids = list(encoded.ids)

        # Truncate if necessary (leave room for generation)
        max_prompt_len = cache_size - max_new_tokens
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        prompt_len = len(prompt_ids)

        # TTT requires sequence length divisible by mini_batch_size (16)
        # Pad prompt to nearest multiple of 16 for prefill
        mini_batch_size = 16
        padded_prompt_len = (
            (prompt_len + mini_batch_size - 1) // mini_batch_size
        ) * mini_batch_size
        pad_len = padded_prompt_len - prompt_len

        # Get pad token (use 0 if not available)
        pad_id = tokenizer.token_to_id("<pad>") or tokenizer.token_to_id("<|pad|>") or 0

        completions = []
        for sample_idx in range(num_samples):
            # Initialize fresh cache for each sample
            cache = model.init_cache(cache_size=cache_size, batch_size=1)

            # Prefill: process entire prompt with left padding for TTT compatibility
            # Left-pad to make length divisible by 16
            padded_ids = [pad_id] * pad_len + prompt_ids
            input_ids = jnp.array(padded_ids, dtype=jnp.int32)[
                None, :
            ]  # [1, padded_prompt_len]
            # Position IDs: 0 for pad tokens, then 0..prompt_len-1 for actual tokens
            position_ids = jnp.concatenate(
                [
                    jnp.zeros(pad_len, dtype=jnp.int32),
                    jnp.arange(prompt_len, dtype=jnp.int32),
                ]
            )[None, :]  # [1, padded_prompt_len]

            logits, cache = jit_prefill(model, input_ids, position_ids, cache)

            # Get logits for last actual token (not padding)
            next_token_logits = logits[0, -1, :]

            # Sample first token
            if temperature > 0:
                probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                key = jax.random.PRNGKey((hash(prompt) + sample_idx * 1000) % (2**31))
                next_token = int(jax.random.categorical(key, jnp.log(probs)))
            else:
                next_token = int(jnp.argmax(next_token_logits))

            generated_tokens = []

            if next_token == eos_id:
                # Empty completion
                pass
            else:
                generated_tokens.append(next_token)
                current_pos = prompt_len

                # Decode: generate tokens one at a time using cache
                for step in range(1, max_new_tokens):
                    if current_pos >= cache_size:
                        break  # Cache full

                    # Single token input
                    input_ids = jnp.array([[next_token]], dtype=jnp.int32)  # [1, 1]
                    position_ids = jnp.array([[current_pos]], dtype=jnp.int32)  # [1, 1]

                    logits, cache = jit_decode(model, input_ids, position_ids, cache)
                    next_token_logits = logits[0, 0, :]

                    # Sample next token
                    if temperature > 0:
                        probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                        key = jax.random.PRNGKey(
                            (hash(prompt) + sample_idx * 1000 + step) % (2**31)
                        )
                        next_token = int(jax.random.categorical(key, jnp.log(probs)))
                    else:
                        next_token = int(jnp.argmax(next_token_logits))

                    if next_token == eos_id:
                        break

                    generated_tokens.append(next_token)
                    current_pos += 1

            # Decode tokens to text
            completion = tokenizer.decode(generated_tokens)

            # Clean up generation artifacts
            # Remove markdown code blocks if present
            if "```" in completion:
                # Extract code from markdown
                parts = completion.split("```")
                if len(parts) >= 2:
                    code_part = parts[0]  # Before first ``` or between ``` markers
                    # If it starts with python, skip that line
                    if code_part.strip().startswith("python"):
                        code_part = "\n".join(code_part.split("\n")[1:])
                    completion = code_part

            # Remove any trailing explanation after function ends
            # Look for return statement followed by double newline
            lines = completion.split("\n")
            clean_lines = []
            for i, line in enumerate(lines):
                clean_lines.append(line)
                # Stop after a line that looks like a return or the function body ends
                if line.strip().startswith("return ") and i > 0:
                    # Check if next non-empty line is unindented (new function/text)
                    remaining = lines[i + 1 :] if i + 1 < len(lines) else []
                    for next_line in remaining:
                        if next_line.strip() == "":
                            continue
                        # If unindented, we've left the function
                        if not next_line.startswith(" ") and not next_line.startswith(
                            "\t"
                        ):
                            break
                        clean_lines.append(next_line)
                    break
            completion = "\n".join(clean_lines)

            completions.append(completion)

        return completions

    return generate


def main():
    args = parse_args()

    # Validate args
    if args.use_ttt and args.no_ttt:
        raise ValueError("Cannot specify both --use_ttt and --no_ttt")

    use_ttt = args.use_ttt and not args.no_ttt

    logger.info(f"=== Gemma 3 {args.model_scale.upper()} HumanEval Evaluation ===")
    logger.info(f"TTT: {'Enabled' if use_ttt else 'Disabled'}")
    logger.info(f"Samples per problem: {args.num_samples}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Chunk size: {args.chunk_size}")

    # Load tokenizer
    tokenizer_name = get_default_tokenizer(args.model_scale)
    logger.info(f"\nTokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)

    # Load model
    logger.info("\nLoading model...")
    model_name = get_model_name(args.model_scale)
    checkpoint_path = args.checkpoint_path or f"hf:google/gemma-3-{args.model_scale}-it"

    model, config = load_ttt_model(
        model_name=model_name,
        fast_weight_type="ttt",
        dtype=jnp.bfloat16,
        seed=args.seed,
        load_pretrained=True,
        checkpoint_path=checkpoint_path,
    )
    model.eval()

    logger.info(f"Model loaded: {model_name}")

    # Load benchmark
    logger.info("\nLoading HumanEval benchmark...")
    benchmark = HumanEvalBenchmark()
    logger.info(f"Loaded {len(benchmark)} problems")

    # Create generate function (includes JIT warmup)
    generate_fn = create_generate_fn(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_ttt=use_ttt,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
    )

    # Evaluate
    logger.info("\nEvaluating...")

    if args.num_problems:
        problems = benchmark.problems[: args.num_problems]
        scores = []
        all_samples = []  # Store samples for debugging

        for problem in tqdm(problems, desc="HumanEval"):
            samples = generate_fn(problem.prompt)
            from ponderttt.evaluation.benchmarks import _check_solution

            passed = any(_check_solution(problem, s) for s in samples)
            scores.append(1.0 if passed else 0.0)

            # Store sample for debugging
            all_samples.append(
                {
                    "task_id": problem.task_id,
                    "prompt": problem.prompt[:200] + "...",  # Truncate for readability
                    "completion": samples[0] if samples else "",
                    "passed": passed,
                }
            )

        results = {
            "pass@1": sum(scores) / len(scores),
            "num_problems": len(scores),
            "correct": int(sum(scores)),
        }
    else:
        results = benchmark.evaluate(generate_fn, k=args.num_samples)
        all_samples = []  # Not available in full eval mode

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ttt_suffix = "ttt" if use_ttt else "baseline"
    output_file = output_dir / f"humaneval_{args.model_scale}_{ttt_suffix}.json"

    full_results = {
        "model_scale": args.model_scale,
        "use_ttt": use_ttt,
        "checkpoint": checkpoint_path,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "chunk_size": args.chunk_size,
        **results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    # Save samples for debugging (if available)
    if all_samples:
        samples_file = output_dir / f"samples_{args.model_scale}_{ttt_suffix}.json"
        with open(samples_file, "w") as f:
            json.dump(all_samples, f, indent=2)
        logger.info(f"Samples saved to: {samples_file}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n=== Results ===")
    logger.info(
        f"Pass@{args.num_samples}: {results.get('pass@k', results.get('pass@1', 0)):.2%}"
    )
    logger.info(f"Problems evaluated: {results.get('num_problems', 0)}")


if __name__ == "__main__":
    main()
