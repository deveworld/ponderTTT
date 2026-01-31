#!/usr/bin/env python3
"""
Evaluate Gemma 3 on HumanEval with and without TTT.

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
from ponderttt.models.gemma3.model import Gemma3TTTModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_default_tokenizer(model_scale: str) -> str:
    """Get default tokenizer for model scale."""
    return f"google/gemma-3-{model_scale}-pt"


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
        default=256,
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
    return parser.parse_args()


def create_generate_fn(
    model: Gemma3TTTModel,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    use_ttt: bool,
    num_samples: int,
):
    """Create generation function for HumanEval."""

    # Get EOS token ID from tokenizer
    eos_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else 1

    def generate(prompt: str) -> list[str]:
        """Generate completions for a prompt."""
        # Tokenize using HuggingFace tokenizer
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = jnp.array(encoded)[None, :]  # [1, seq_len]

        # Generate
        completions = []
        for sample_idx in range(num_samples):
            generated_ids = input_ids

            for step in range(max_new_tokens):
                # Forward pass
                position_ids = jnp.arange(generated_ids.shape[1])[None, :]
                attention_mask = jnp.ones_like(generated_ids)

                outputs = model(
                    generated_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_ttt=use_ttt,
                )

                logits = outputs["logits"]
                next_token_logits = logits[:, -1, :]

                # Sample or greedy
                if temperature > 0:
                    probs = jax.nn.softmax(next_token_logits / temperature, axis=-1)
                    # Use different key for each sample and step
                    key = jax.random.PRNGKey(
                        (hash(prompt) + sample_idx * 1000 + step) % (2**31)
                    )
                    next_token = jax.random.categorical(key, jnp.log(probs))
                else:
                    next_token = jnp.argmax(next_token_logits, axis=-1)

                next_token = next_token.reshape(1, 1)
                generated_ids = jnp.concatenate([generated_ids, next_token], axis=1)

                # Check for EOS
                if int(next_token[0, 0]) == eos_id:
                    break

            # Decode only the new tokens
            new_ids = generated_ids[0, input_ids.shape[1] :].tolist()
            completion = tokenizer.decode(new_ids, skip_special_tokens=True)

            # Stop at first double newline (HumanEval convention for end of function)
            if "\n\n" in completion:
                completion = completion.split("\n\n")[0]

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

    # Load tokenizer first (needed for model config)
    tokenizer_name = get_default_tokenizer(args.model_scale)
    logger.info(f"\nTokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)

    # Load model
    logger.info("\nLoading model...")
    checkpoint = args.checkpoint_path or f"hf:google/gemma-3-{args.model_scale}-pt"

    model = Gemma3TTTModel.from_pretrained(
        checkpoint,
        model_scale=args.model_scale,
        rngs=nnx.Rngs(0),
    )
    model.eval()

    logger.info(f"Model loaded: {args.model_scale}")

    # Load benchmark
    logger.info("\nLoading HumanEval benchmark...")
    benchmark = HumanEvalBenchmark()
    logger.info(f"Loaded {len(benchmark)} problems")

    # Create generate function
    generate_fn = create_generate_fn(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_ttt=use_ttt,
        num_samples=args.num_samples,
    )

    # Evaluate
    logger.info("\nEvaluating...")

    if args.num_problems:
        # Evaluate subset
        problems = benchmark.problems[: args.num_problems]
        scores = []

        for problem in tqdm(problems, desc="HumanEval"):
            samples = generate_fn(problem.prompt)
            # For single sample, just check if it passes
            from ponderttt.evaluation.benchmarks import _check_solution

            passed = any(_check_solution(problem, s) for s in samples)
            scores.append(1.0 if passed else 0.0)

        results = {
            "pass@1": sum(scores) / len(scores),
            "num_problems": len(scores),
            "correct": int(sum(scores)),
        }
    else:
        results = benchmark.evaluate(generate_fn, k=args.num_samples)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ttt_suffix = "ttt" if use_ttt else "baseline"
    output_file = output_dir / f"humaneval_{args.model_scale}_{ttt_suffix}.json"

    full_results = {
        "model_scale": args.model_scale,
        "use_ttt": use_ttt,
        "checkpoint": checkpoint,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        **results,
    }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("\n=== Results ===")
    logger.info(
        f"Pass@{args.num_samples}: {results.get('pass@k', results.get('pass@1', 0)):.2%}"
    )
    logger.info(f"Problems evaluated: {results.get('num_problems', 0)}")


if __name__ == "__main__":
    main()
