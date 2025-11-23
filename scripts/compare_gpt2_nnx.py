"""
Compare GPT-2 Flax NNX logits to Hugging Face Transformers logits.

Usage:
    python scripts/compare_gpt2_nnx.py --text "Hello world"

Notes:
- Requires `transformers` (PyTorch) and the GPT-2 checkpoint available locally or
  downloadable from the Hugging Face Hub.
- Uses the same tokenizer for both models to ensure token alignment.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import numpy as np

from ponderttt.models import load_ttt_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPT-2 NNX vs Transformers logits")
    parser.add_argument("--model_name", type=str, default="gpt2", help="GPT-2 variant")
    parser.add_argument("--text", type=str, default="Hello world", help="Input text")
    parser.add_argument("--max_length", type=int, default=32, help="Max tokens to compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] transformers/torch not available: {exc}")
        return

    print("=" * 60)
    print(f"Comparing GPT-2 ({args.model_name}) NNX vs Transformers")
    print("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids_pt = inputs["input_ids"]

    # Transformers (PyTorch) reference
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    hf_model.eval()
    with torch.no_grad():
        logits_pt = hf_model(input_ids_pt).logits.cpu().numpy()

    # NNX model (uses HF weights)
    model_nnx, _ = load_ttt_model(
        model_name=args.model_name,
        fast_weight_type="ttt",
        load_pretrained=True,
        vocab_size=tokenizer.vocab_size,
    )
    model_nnx.eval()

    input_ids_nnx = jnp.array(input_ids_pt.numpy(), dtype=jnp.int32)
    outputs_nnx = cast(dict[str, Any], model_nnx(input_ids_nnx, use_ttt=False))
    logits_nnx = np.array(outputs_nnx["logits"])

    # Compare
    min_len = min(logits_pt.shape[1], logits_nnx.shape[1], args.max_length)
    logits_pt = logits_pt[:, :min_len]
    logits_nnx = logits_nnx[:, :min_len]

    mse = float(np.mean((logits_pt - logits_nnx) ** 2))
    max_abs = float(np.max(np.abs(logits_pt - logits_nnx)))

    print(f"Input text: {args.text!r}")
    print(f"Sequence length compared: {min_len}")
    print(f"MSE between logits: {mse:.6f}")
    print(f"Max abs diff: {max_abs:.6f}")

    # Show top-5 tokens for first position
    def top_k(logits: np.ndarray, k: int = 5):
        top_idx = np.argsort(logits)[..., ::-1][0, 0, :k]
        return [(idx, float(logits[0, 0, idx])) for idx in top_idx]

    print("\nTop-5 tokens at position 0 (Transformers):")
    for idx, val in top_k(logits_pt):
        print(f"  {idx}: {val:.4f}")

    print("Top-5 tokens at position 0 (NNX):")
    for idx, val in top_k(logits_nnx):
        print(f"  {idx}: {val:.4f}")


if __name__ == "__main__":
    main()
