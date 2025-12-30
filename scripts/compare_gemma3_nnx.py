"""
Compare Gemma 3 NNX logits to Hugging Face Transformers logits.

Usage:
    python scripts/compare_gemma3_nnx.py --text "Hello world"
    python scripts/compare_gemma3_nnx.py --model_scale 4b --text "def hello():"

Notes:
- Requires `transformers` (PyTorch) with Gemma 3 support.
- Uses the same tokenizer for both models to ensure token alignment.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Gemma 3 NNX vs Transformers logits"
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="4b",
        choices=["4b", "12b"],
        help="Gemma 3 model scale",
    )
    parser.add_argument("--text", type=str, default="Hello world", help="Input text")
    parser.add_argument(
        "--max_length", type=int, default=32, help="Max tokens to compare"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Override checkpoint path (default: hf:google/gemma-3-{scale}-pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:
        print(f"[FAIL] transformers/torch not available: {exc}")
        return

    # Model names
    hf_model_name = f"google/gemma-3-{args.model_scale}-pt"
    checkpoint_path = args.checkpoint_path or f"hf:{hf_model_name}"

    print("=" * 60)
    print(f"Comparing Gemma 3 ({args.model_scale}) NNX vs Transformers")
    print("=" * 60)

    # Tokenizer (use Gemma 2 tokenizer for compatibility)
    tokenizer_name = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids_pt = inputs["input_ids"]
    print(f"Input: {args.text!r}")
    print(f"Token IDs: {input_ids_pt.tolist()}")

    # Transformers (PyTorch) reference
    print(f"\nLoading HuggingFace model: {hf_model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_model.eval()
    with torch.no_grad():
        outputs_hf = hf_model(input_ids_pt.to(hf_model.device))
        logits_pt = outputs_hf.logits.float().cpu().numpy()
    print(f"HF logits shape: {logits_pt.shape}")
    print(f"HF logits range: [{logits_pt.min():.2f}, {logits_pt.max():.2f}]")

    # NNX model
    print(f"\nLoading NNX model: {args.model_scale}")
    from ponderttt.models import load_ttt_model

    model_nnx, _ = load_ttt_model(
        model_name=f"gemma3-{args.model_scale}",
        fast_weight_type="ttt",
        load_pretrained=True,
        checkpoint_path=checkpoint_path,
        dtype=jnp.bfloat16,
    )
    model_nnx.eval()

    input_ids_nnx = jnp.array(input_ids_pt.numpy(), dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids_nnx)
    position_ids = jnp.arange(input_ids_nnx.shape[1], dtype=jnp.int32)[None, :]

    # Gemma 3 returns (output_dict, cache)
    outputs_nnx, _ = model_nnx(
        input_ids_nnx,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits_nnx = np.array(outputs_nnx["logits"], dtype=np.float32)
    print(f"NNX logits shape: {logits_nnx.shape}")
    print(f"NNX logits range: [{logits_nnx.min():.2f}, {logits_nnx.max():.2f}]")

    # Compare
    min_len = min(logits_pt.shape[1], logits_nnx.shape[1], args.max_length)
    logits_pt = logits_pt[:, :min_len]
    logits_nnx = logits_nnx[:, :min_len]

    mse = float(np.mean((logits_pt - logits_nnx) ** 2))
    max_abs = float(np.max(np.abs(logits_pt - logits_nnx)))

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Sequence length compared: {min_len}")
    print(f"MSE between logits: {mse:.6f}")
    print(f"Max abs diff: {max_abs:.6f}")

    if mse < 1.0:
        print("\n✅ PASS: Logits match closely")
    elif mse < 10.0:
        print("\n⚠️ WARNING: Small difference in logits (possibly dtype/precision)")
    else:
        print("\n❌ FAIL: Logits differ significantly")

    # Show top-5 tokens for first position
    def top_k(logits: np.ndarray, k: int = 5):
        top_idx = np.argsort(logits)[..., ::-1][0, 0, :k]
        return [(int(idx), float(logits[0, 0, idx])) for idx in top_idx]

    print("\nTop-5 tokens at position 0 (Transformers):")
    for idx, val in top_k(logits_pt):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    print("\nTop-5 tokens at position 0 (NNX):")
    for idx, val in top_k(logits_nnx):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    # Compute cross-entropy loss for reference
    print("\n" + "=" * 60)
    print("LOSS COMPUTATION")
    print("=" * 60)

    # Simple CE loss calculation
    targets = input_ids_pt[:, 1:].numpy()
    logits_for_loss = logits_nnx[:, :-1, :]

    # Softmax
    log_probs = logits_for_loss - np.log(
        np.sum(np.exp(logits_for_loss), axis=-1, keepdims=True)
    )
    # Gather
    ce_loss = -log_probs[0, np.arange(len(targets[0])), targets[0]]
    avg_loss = float(np.mean(ce_loss))

    print(f"Cross-entropy loss (NNX, no TTT): {avg_loss:.4f}")
    if avg_loss < 5.0:
        print("✅ Loss looks reasonable")
    else:
        print("❌ Loss is too high - model may have issues")


if __name__ == "__main__":
    main()
