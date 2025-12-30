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
        choices=["1b", "4b", "12b", "27b"],
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

    # Tokenizer
    tokenizer_name = f"google/gemma-3-{args.model_scale}-pt"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        print(
            f"Warning: Could not load {tokenizer_name}, falling back to google/gemma-3-4b-pt"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")

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
        dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_model.eval()
    with torch.no_grad():
        outputs_hf = hf_model(input_ids_pt.to(hf_model.device))
        logits_pt = outputs_hf.logits.float().cpu().numpy()
    print(f"HF logits shape: {logits_pt.shape}")
    print(f"HF model structure: {hf_model}")
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
    print(
        f"NNX embedding shape: {model_nnx.base_model.embedder.input_embedding[...].shape}"
    )
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

    # Show top-5 tokens
    def top_k(logits: np.ndarray, pos: int, k: int = 5):
        top_idx = np.argsort(logits)[..., ::-1][0, pos, :k]
        return [(int(idx), float(logits[0, pos, idx])) for idx in top_idx]

    print("\nTop-5 tokens at position 0 (Transformers):")
    for idx, val in top_k(logits_pt, 0):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    print("\nTop-5 tokens at position 0 (NNX):")
    for idx, val in top_k(logits_nnx, 0):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    # Show top-5 tokens for last position
    last_pos = min_len - 1
    print(f"\nTop-5 tokens at position {last_pos} (Transformers):")
    for idx, val in top_k(logits_pt, last_pos):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    print(f"\nTop-5 tokens at position {last_pos} (NNX):")
    for idx, val in top_k(logits_nnx, last_pos):
        token = tokenizer.decode([idx])
        print(f"  {idx:6d}: {val:8.4f}  {token!r}")

    # Compute cross-entropy loss for reference
    print("\n" + "=" * 60)
    print("LOSS COMPUTATION")
    print("=" * 60)

    # Compute cross-entropy loss for reference
    print("\n" + "=" * 60)
    print("LOSS COMPUTATION")
    print("=" * 60)

    # Simple CE loss calculation
    # We predict the next token, so targets are input_ids[1:]
    targets = input_ids_pt[:, 1:].numpy()

    def compute_loss(logits, targets):
        # logits: [batch, seq_len, vocab] -> we need [:, :-1, :] to match targets
        logits_for_loss = logits[:, :-1, :]

        # Softmax for probabilities
        # Safety: subtract max for stability
        logits_max = np.max(logits_for_loss, axis=-1, keepdims=True)
        logits_stable = logits_for_loss - logits_max
        exp_logits = np.exp(logits_stable)
        log_probs = logits_stable - np.log(np.sum(exp_logits, axis=-1, keepdims=True))

        # Gather log probs for target tokens
        batch_indices = np.arange(targets.shape[0])[:, None]
        seq_indices = np.arange(targets.shape[1])[None, :]
        target_log_probs = log_probs[batch_indices, seq_indices, targets]

        return -np.mean(target_log_probs)

    loss_ref = compute_loss(logits_pt, targets)
    loss_nnx = compute_loss(logits_nnx, targets)

    print(f"Cross-entropy loss (Transformers): {loss_ref:.4f}")
    print(f"Cross-entropy loss (NNX, no TTT): {loss_nnx:.4f}")

    loss_diff = abs(loss_ref - loss_nnx)
    loss_rel_diff = loss_diff / (abs(loss_ref) + 1e-6)

    print(f"Loss Difference: {loss_diff:.4f}")
    print(f"Relative Difference: {loss_rel_diff:.4%}")

    # Use 1.5% relative tolerance for BF16 models (Gemma 3 has soft-capping etc which adds noise)
    if loss_rel_diff < 0.015:
        print("✅ PASS: Losses match (within 1.5% tolerance)")
    else:
        print("❌ FAIL: Losses differ significantly")


if __name__ == "__main__":
    main()
