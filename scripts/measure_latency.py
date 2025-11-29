#!/usr/bin/env python3
"""
Measure wall-clock latency for Hard Skip (Binary Gating).

Compares:
- SKIP (baseline): No TTT
- UPDATE_1 (fixed): Always run TTT
- Hard Skip (Ours): Conditional TTT based on binary decision

Usage:
    python scripts/measure_latency.py --checkpoint outputs/hard_skip/125m_skip0.8/checkpoint_XXXX
"""

import argparse
import time
import jax
import jax.numpy as jnp
from flax import nnx
import pandas as pd

from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import BinaryGatingConfig, BinaryGatingNetwork
from ponderttt.utils import FeatureExtractor
from ponderttt.utils.checkpointing import load_checkpoint

MODEL_SCALES = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}


class TrainableSystem(nnx.Module):
    """Mirror of TrainableSystem from train_hard_skip.py for checkpoint loading."""
    def __init__(self, ttt_model, gating_net):
        self.fast_layer = ttt_model.fast_layer
        self.fast_norm = ttt_model.fast_norm
        self.gating_net = gating_net
        if hasattr(ttt_model, 'lm_head'):
            self.lm_head = ttt_model.lm_head
        else:
            self.lm_head = None


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Latency")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Binary Gating checkpoint",
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="125m",
        choices=["125m", "350m", "1b"],
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of timing trials",
    )
    return parser.parse_args()


def measure_latency(func, num_warmup, num_trials):
    """Measure latency of a function."""
    # Warmup
    for _ in range(num_warmup):
        func()
        jax.block_until_ready(func())

    # Timing
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = func()
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)


def main():
    args = parse_args()

    print(f"Benchmarking Latency: {args.model_scale}, Chunk={args.chunk_size}")

    # Load model
    model_name = MODEL_SCALES[args.model_scale]
    ttt_model, _ = load_ttt_model(model_name, load_pretrained=True)

    # Initialize Binary Gating network
    config = BinaryGatingConfig(feature_dim=32, hidden_dim=64)
    rngs = nnx.Rngs(0)
    gating_net = BinaryGatingNetwork(config, rngs)

    # Create TrainableSystem for checkpoint loading
    trainable_system = TrainableSystem(ttt_model, gating_net)

    # Load checkpoint (model state only, no optimizer needed for inference)
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = load_checkpoint(args.checkpoint, target=None)

    # Update model with loaded state
    if "state" in ckpt and "model" in ckpt["state"]:
        nnx.update(trainable_system, ckpt["state"]["model"])
    print(f"Checkpoint loaded (step {ckpt.get('step', 'unknown')})")

    # Feature extractor
    from ponderttt.data import get_tokenizer as get_tok
    tok = get_tok(model_name)
    feature_extractor = FeatureExtractor(
        vocab_size=tok.get_vocab_size(),
        pad_token_id=tok.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Create dummy input
    dummy_input = jnp.ones((1, args.chunk_size), dtype=jnp.int32)

    # JIT compile functions
    print("Compiling functions...")

    @jax.jit
    def forward_skip(x):
        return ttt_model(x, use_ttt=False, gating_scale=None)

    @jax.jit
    def forward_update(x):
        return ttt_model(x, use_ttt=True, gating_scale=[[1.0]])

    def get_features_and_decision(x):
        """Get features from base model output and make gating decision."""
        out = forward_skip(x)
        features = feature_extractor.extract(
            input_ids=x,
            logits=out["logits"],
            attention_mask=jnp.ones_like(x),
            budget_remaining=1.0,
        )
        return gating_net.get_decision(features)

    # Warmup all functions
    print("Warmup...")
    for _ in range(args.num_warmup):
        forward_skip(dummy_input)
        forward_update(dummy_input)
        get_features_and_decision(dummy_input)

    jax.block_until_ready(forward_skip(dummy_input))

    # Measure SKIP baseline
    latency_skip = measure_latency(
        lambda: forward_skip(dummy_input),
        args.num_warmup,
        args.num_trials,
    )

    # Measure UPDATE_1 (fixed)
    latency_update = measure_latency(
        lambda: forward_update(dummy_input),
        args.num_warmup,
        args.num_trials,
    )

    # Measure gating overhead only (includes base model forward for features)
    latency_gating = measure_latency(
        lambda: get_features_and_decision(dummy_input),
        args.num_warmup,
        args.num_trials,
    )

    # Measure Hard Skip with actual decisions
    # Run multiple trials to get average decision rate
    total_skip = 0
    total_update = 0
    decision_times = []

    for _ in range(args.num_trials):
        _, decision = get_features_and_decision(dummy_input)
        decision_val = int(decision[0])

        start = time.perf_counter()
        if decision_val == 0:  # SKIP
            # SKIP: No additional computation (base forward already done in gating)
            total_skip += 1
        else:  # UPDATE
            result = forward_update(dummy_input)
            jax.block_until_ready(result)
            total_update += 1
        end = time.perf_counter()

        decision_times.append((end - start) * 1000 + latency_gating)

    latency_hard_skip = sum(decision_times) / len(decision_times)
    skip_rate = total_skip / (total_skip + total_update)
    update_rate = total_update / (total_skip + total_update)

    # Expected latency based on skip rate
    expected_latency = skip_rate * latency_skip + update_rate * latency_update + latency_gating

    # Results
    results = [
        {"Method": "SKIP (Baseline)", "Latency (ms)": latency_skip, "Rel. Speed": 1.0, "Note": ""},
        {"Method": "UPDATE_1 (Fixed)", "Latency (ms)": latency_update, "Rel. Speed": latency_update / latency_skip, "Note": ""},
        {"Method": "Gating Overhead", "Latency (ms)": latency_gating, "Rel. Speed": latency_gating / latency_skip, "Note": ""},
        {"Method": "Hard Skip (Ours)", "Latency (ms)": latency_hard_skip, "Rel. Speed": latency_hard_skip / latency_skip, "Note": f"Skip={skip_rate:.0%}"},
        {"Method": "Expected (Theory)", "Latency (ms)": expected_latency, "Rel. Speed": expected_latency / latency_skip, "Note": f"Skip={skip_rate:.0%}"},
    ]

    df = pd.DataFrame(results)
    print(f"\n=== Latency Results (ms per chunk) ===")
    print(df.to_string(index=False))

    speedup = latency_update / latency_hard_skip
    print(f"\nSpeedup vs UPDATE_1: {speedup:.2f}x")

    # Cost model verification
    theoretical_cost = 1.0 + 2.0 * update_rate  # 1 + 2λ
    actual_cost = latency_hard_skip / latency_skip
    print(f"\nCost Model Verification:")
    print(f"  Skip Rate: {skip_rate:.1%}")
    print(f"  Update Rate: {update_rate:.1%}")
    print(f"  Theoretical Cost (1 + 2λ): {theoretical_cost:.2f}x")
    print(f"  Actual Latency Ratio: {actual_cost:.2f}x")

    # LaTeX table
    print("\n" + "=" * 70)
    print("FOR PAPER (LaTeX)")
    print("=" * 70)
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Wall-clock latency per {args.chunk_size}-token chunk on GPU (Hard Skip).}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Latency (ms)}} & \\textbf{{Rel. Speed}} & \\textbf{{Speedup vs UPDATE\\_1}} \\\\
\\midrule
Baseline (SKIP) & {latency_skip:.2f} & 1.00x & - \\\\
Baseline (UPDATE\\_1) & {latency_update:.2f} & {latency_update/latency_skip:.2f}x & 1.00x \\\\
\\midrule
\\textbf{{PonderTTT (Ours)}} & \\textbf{{{latency_hard_skip:.2f}}} & \\textbf{{{latency_hard_skip/latency_skip:.2f}x}} & \\textbf{{{speedup:.2f}x}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    main()
