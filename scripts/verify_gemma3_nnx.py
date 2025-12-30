"""
Verify Gemma 3 NNX model outputs are reasonable (no HuggingFace comparison).

Usage:
    python scripts/verify_gemma3_nnx.py --model_scale 4b

Checks:
1. Logits are in reasonable range (not NaN/Inf)
2. Cross-entropy loss is reasonable (<10 for pretrained model)
3. Model can run forward pass without errors
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Gemma 3 NNX model")
    parser.add_argument(
        "--model_scale",
        type=str,
        default="4b",
        choices=["4b", "12b"],
        help="Gemma 3 model scale",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Override checkpoint path"
    )
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f"Verifying Gemma 3 ({args.model_scale}) NNX Model")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")

    # Load tokenizer
    from ponderttt.data import get_tokenizer

    tokenizer = get_tokenizer("google/gemma-2-2b")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Load model
    checkpoint_path = args.checkpoint_path or f"hf:google/gemma-3-{args.model_scale}-pt"
    print(f"\nLoading model from: {checkpoint_path}")

    from ponderttt.models import load_ttt_model

    model, _ = load_ttt_model(
        model_name=f"gemma3-{args.model_scale}",
        fast_weight_type="ttt",
        load_pretrained=True,
        checkpoint_path=checkpoint_path,
        dtype=jnp.bfloat16,
    )
    model.eval()
    print("Model loaded successfully!")

    # Create test input
    test_text = "def hello_world():\n    print('Hello, world!')\n"
    encoded = tokenizer.encode(test_text)
    input_ids = jnp.array([encoded.ids[: args.seq_length]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.arange(input_ids.shape[1], dtype=jnp.int32)[None, :]

    print(f"\nTest input: {test_text!r}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs (first 10): {input_ids[0, :10].tolist()}")

    # Forward pass WITHOUT TTT
    print("\n--- Forward pass (use_ttt=False) ---")
    outputs, _ = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=False,
    )
    logits = outputs["logits"]

    logits_np = np.array(logits, dtype=np.float32)
    print(f"Logits shape: {logits_np.shape}")
    print(f"Logits range: [{logits_np.min():.2f}, {logits_np.max():.2f}]")
    print(f"Logits mean: {logits_np.mean():.4f}")
    print(f"Logits std: {logits_np.std():.4f}")
    print(f"Has NaN: {np.isnan(logits_np).any()}")
    print(f"Has Inf: {np.isinf(logits_np).any()}")

    # Compute cross-entropy loss
    targets = input_ids[:, 1:]
    logits_for_loss = logits_np[:, :-1, :]

    # Numerically stable softmax
    max_logits = logits_for_loss.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(
        np.sum(np.exp(logits_for_loss - max_logits), axis=-1, keepdims=True)
    )
    log_probs = logits_for_loss - max_logits - log_sum_exp

    # Gather target log probs
    targets_np = np.array(targets)
    ce_losses = []
    for b in range(targets_np.shape[0]):
        for t in range(targets_np.shape[1]):
            ce_losses.append(-log_probs[b, t, targets_np[b, t]])

    avg_loss_no_ttt = float(np.mean(ce_losses))
    print(f"\nCross-entropy loss (no TTT): {avg_loss_no_ttt:.4f}")

    # Forward pass WITH TTT
    print("\n--- Forward pass (use_ttt=True) ---")
    outputs_ttt, _ = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_ttt=True,
    )
    logits_ttt = outputs_ttt["logits"]
    ttt_stats = outputs_ttt.get("ttt_stats", {})

    logits_ttt_np = np.array(logits_ttt, dtype=np.float32)
    print(f"Logits (TTT) range: [{logits_ttt_np.min():.2f}, {logits_ttt_np.max():.2f}]")

    if ttt_stats:
        print(f"TTT stats keys: {list(ttt_stats.keys())}")
        for k, v in ttt_stats.items():
            if hasattr(v, "mean"):
                print(f"  {k}: {float(v.mean()):.4f}")
            else:
                print(f"  {k}: {v}")

    # Compute loss with TTT
    logits_ttt_for_loss = logits_ttt_np[:, :-1, :]
    max_logits_ttt = logits_ttt_for_loss.max(axis=-1, keepdims=True)
    log_sum_exp_ttt = np.log(
        np.sum(np.exp(logits_ttt_for_loss - max_logits_ttt), axis=-1, keepdims=True)
    )
    log_probs_ttt = logits_ttt_for_loss - max_logits_ttt - log_sum_exp_ttt

    ce_losses_ttt = []
    for b in range(targets_np.shape[0]):
        for t in range(targets_np.shape[1]):
            ce_losses_ttt.append(-log_probs_ttt[b, t, targets_np[b, t]])

    avg_loss_ttt = float(np.mean(ce_losses_ttt))
    print(f"Cross-entropy loss (with TTT): {avg_loss_ttt:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    checks_passed = 0
    total_checks = 4

    # Check 1: No NaN/Inf
    if not np.isnan(logits_np).any() and not np.isinf(logits_np).any():
        print("✅ Check 1: No NaN/Inf in logits")
        checks_passed += 1
    else:
        print("❌ Check 1: NaN/Inf detected in logits")

    # Check 2: Reasonable logit range
    if -100 < logits_np.min() and logits_np.max() < 100:
        print("✅ Check 2: Logits in reasonable range")
        checks_passed += 1
    else:
        print("❌ Check 2: Logits out of range")

    # Check 3: Reasonable loss (pretrained model should have loss < 10)
    if avg_loss_no_ttt < 10:
        print(f"✅ Check 3: Loss reasonable ({avg_loss_no_ttt:.2f} < 10)")
        checks_passed += 1
    else:
        print(f"❌ Check 3: Loss too high ({avg_loss_no_ttt:.2f} >= 10)")

    # Check 4: TTT improves or maintains loss
    if avg_loss_ttt <= avg_loss_no_ttt + 0.5:
        print(f"✅ Check 4: TTT loss reasonable ({avg_loss_ttt:.2f})")
        checks_passed += 1
    else:
        print(
            f"❌ Check 4: TTT loss degraded ({avg_loss_ttt:.2f} > {avg_loss_no_ttt:.2f})"
        )

    print(f"\nPassed {checks_passed}/{total_checks} checks")

    if checks_passed == total_checks:
        print("\n🎉 All checks passed! Model is working correctly.")
    else:
        print("\n⚠️ Some checks failed. Further debugging needed.")


if __name__ == "__main__":
    main()
