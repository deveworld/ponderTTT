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
        default=512,
        help="Fixed chunk size for JIT compilation (must match training)",
    )
    return parser.parse_args()


def make_forward_fn(use_ttt: bool):
    """Create forward function for JIT compilation.

    This follows the pattern from compare_methods.py - create the function
    once with fixed signature, then JIT compile.
    """

    def forward_fn(
        model: Any,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array,
    ) -> jax.Array:
        """Forward pass returning logits."""
        result = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_ttt=use_ttt,
        )
        # Handle tuple return (Gemma3) vs dict return
        if isinstance(result, tuple):
            outputs = result[0]
        else:
            outputs = result
        return outputs["logits"]

    return forward_fn


def create_generate_fn(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    use_ttt: bool,
    num_samples: int,
    chunk_size: int,
):
    """Create generation function for HumanEval with fixed chunk sizes."""

    # Get EOS token ID
    eos_id = (
        tokenizer.token_to_id("<eos>")
        or tokenizer.token_to_id("</s>")
        or tokenizer.token_to_id("<end_of_turn>")
        or 1
    )
    pad_id = tokenizer.token_to_id("<|pad|>") or 0

    # Create and JIT compile the forward function (same pattern as compare_methods.py)
    forward_fn = make_forward_fn(use_ttt)
    jit_forward = nnx.jit(forward_fn)

    # Warmup with fixed chunk size
    logger.info(f"Warming up JIT with chunk_size={chunk_size}...")
    dummy_ids = jnp.ones((1, chunk_size), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, chunk_size), dtype=jnp.int32)
    dummy_pos = jnp.arange(chunk_size, dtype=jnp.int32)[None, :]
    _ = jit_forward(model, dummy_ids, dummy_mask, dummy_pos)
    logger.info("JIT warmup complete.")

    def generate(prompt: str) -> list[str]:
        """Generate completions using fixed-size chunked inference."""
        # Apply chat template for IT models
        formatted_prompt = format_prompt_for_it_model(prompt)

        # Tokenize the formatted prompt
        encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        prompt_ids = list(encoded.ids)

        # Truncate if necessary
        max_prompt_len = chunk_size - max_new_tokens
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        completions = []
        for sample_idx in range(num_samples):
            generated = list(prompt_ids)

            for step in range(max_new_tokens):
                # Prepare fixed-size input with left padding
                seq_len = len(generated)
                if seq_len < chunk_size:
                    # Pad to chunk_size
                    pad_len = chunk_size - seq_len
                    padded_ids = [pad_id] * pad_len + generated
                    mask = [0] * pad_len + [1] * seq_len
                else:
                    # Truncate from left (keep last chunk_size tokens)
                    padded_ids = generated[-chunk_size:]
                    mask = [1] * chunk_size

                # Convert to JAX arrays with fixed shape
                input_ids = jnp.array(padded_ids, dtype=jnp.int32)[None, :]
                attention_mask = jnp.array(mask, dtype=jnp.int32)[None, :]
                position_ids = jnp.arange(chunk_size, dtype=jnp.int32)[None, :]

                # Forward pass (JIT compiled, fixed shape = no recompilation)
                logits = jit_forward(model, input_ids, attention_mask, position_ids)

                # Get logits for last actual token position
                last_pos = chunk_size - 1  # Always last position with left padding
                next_token_logits = logits[0, last_pos, :]

                # Sample or greedy
                if temperature > 0:
                    probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                    key = jax.random.PRNGKey(
                        (hash(prompt) + sample_idx * 1000 + step) % (2**31)
                    )
                    next_token = int(jax.random.categorical(key, jnp.log(probs)))
                else:
                    next_token = int(jnp.argmax(next_token_logits))

                # Check for EOS
                if next_token == eos_id:
                    break

                generated.append(next_token)

            # Decode only the new tokens
            new_ids = generated[len(prompt_ids) :]
            completion = tokenizer.decode(new_ids)

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
