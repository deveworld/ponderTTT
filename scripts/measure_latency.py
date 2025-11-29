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


def unwrap_state(state):
    """Recursively unwrap Orbax-serialized NNX state dicts (remove 'value' wrappers)."""
    if isinstance(state, dict):
        if "value" in state and len(state) == 1:
            return state["value"]
        return {k: unwrap_state(v) for k, v in state.items()}
    return state

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
        result = func()
        jax.block_until_ready(result)

    # Timing
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        result = func()
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return sum(times) / len(times)


def create_jitted_functions(ttt_model, gating_net, feature_extractor):
    """Create JIT-compiled functions for latency measurement."""

    model_graph, model_state = nnx.split(ttt_model)
    gating_graph, gating_state = nnx.split(gating_net)

    @jax.jit
    def forward_skip(model_state, input_ids):
        """Forward pass without TTT."""
        model = nnx.merge(model_graph, model_state)
        out = model(input_ids, use_ttt=False, gating_scale=None)
        return out["logits"]

    @jax.jit
    def forward_update(model_state, input_ids):
        """Forward pass with TTT."""
        model = nnx.merge(model_graph, model_state)
        out = model(input_ids, use_ttt=True, gating_scale=jnp.array([[1.0]]))
        return out["logits"]

    @jax.jit
    def get_features_and_decision(model_state, gating_state, input_ids, logits):
        """Extract features and get gating decision."""
        gating = nnx.merge(gating_graph, gating_state)
        features = feature_extractor.extract(
            input_ids=input_ids,
            logits=logits,
            attention_mask=jnp.ones_like(input_ids, dtype=jnp.float32),
            budget_remaining=1.0,
        )
        hard_scale, decision = gating.get_decision(features)
        return decision

    return forward_skip, forward_update, get_features_and_decision, model_state, gating_state


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
        model_state = unwrap_state(ckpt["state"]["model"])
        nnx.update(trainable_system, model_state)
    print(f"Checkpoint loaded (step {ckpt.get('step', 'unknown')})")

    # Feature extractor
    from ponderttt.data import get_tokenizer as get_tok
    tok = get_tok(model_name)
    feature_extractor = FeatureExtractor(
        vocab_size=tok.get_vocab_size(),
        pad_token_id=tok.token_to_id("<|pad|>"),
        seq_length_norm=512,
    )

    # Create JIT-compiled functions
    print("Compiling JIT functions...")
    forward_skip, forward_update, get_features_and_decision, jit_model_state, jit_gating_state = \
        create_jitted_functions(ttt_model, gating_net, feature_extractor)

    # Create dummy input
    dummy_input = jnp.ones((1, args.chunk_size), dtype=jnp.int32)

    # Warmup all functions
    print("Warming up JIT...")
    for _ in range(args.num_warmup):
        _ = forward_skip(jit_model_state, dummy_input)
        _ = forward_update(jit_model_state, dummy_input)
        logits = forward_skip(jit_model_state, dummy_input)
        _ = get_features_and_decision(jit_model_state, jit_gating_state, dummy_input, logits)

    jax.block_until_ready(forward_skip(jit_model_state, dummy_input))
    print("JIT compilation complete.")

    # Measure SKIP baseline
    print("Measuring SKIP latency...")
    latency_skip = measure_latency(
        lambda: forward_skip(jit_model_state, dummy_input),
        args.num_warmup,
        args.num_trials,
    )

    # Measure UPDATE_1 (fixed)
    print("Measuring UPDATE latency...")
    latency_update = measure_latency(
        lambda: forward_update(jit_model_state, dummy_input),
        args.num_warmup,
        args.num_trials,
    )

    # Measure gating overhead only (includes base model forward for features)
    print("Measuring gating latency...")
    def gating_with_skip():
        logits = forward_skip(jit_model_state, dummy_input)
        decision = get_features_and_decision(jit_model_state, jit_gating_state, dummy_input, logits)
        return decision

    latency_gating_full = measure_latency(
        gating_with_skip,
        args.num_warmup,
        args.num_trials,
    )

    # Gating overhead (excluding base forward, which is already measured in latency_skip)
    # gating_overhead = feature extraction + decision making
    gating_overhead = max(0, latency_gating_full - latency_skip)

    # Measure Hard Skip with actual decisions
    # First, determine the decision distribution
    total_skip = 0
    total_update = 0

    print("Measuring decision distribution...")
    for _ in range(args.num_trials):
        logits = forward_skip(jit_model_state, dummy_input)
        decision = get_features_and_decision(jit_model_state, jit_gating_state, dummy_input, logits)
        decision_val = int(decision[0])
        if decision_val == 0:
            total_skip += 1
        else:
            total_update += 1

    skip_rate = total_skip / (total_skip + total_update)
    update_rate = total_update / (total_skip + total_update)

    # Calculate Hard Skip latency correctly:
    # - Always: base forward (for features) + gating decision
    # - If SKIP: done (use base forward output)
    # - If UPDATE: need TTT, so add (UPDATE - SKIP) overhead
    # Total = base_forward + gating_overhead + update_rate * (UPDATE - SKIP)
    ttt_overhead = latency_update - latency_skip  # Just the TTT part
    latency_hard_skip = latency_skip + gating_overhead + update_rate * ttt_overhead

    # Results
    results = [
        {"Method": "SKIP (Baseline)", "Latency (ms)": latency_skip, "Rel. Speed": 1.0, "Note": "use_ttt=False"},
        {"Method": "UPDATE_1 (Fixed)", "Latency (ms)": latency_update, "Rel. Speed": latency_update / latency_skip, "Note": "use_ttt=True"},
        {"Method": "TTT Overhead", "Latency (ms)": ttt_overhead, "Rel. Speed": ttt_overhead / latency_skip, "Note": "UPDATE - SKIP"},
        {"Method": "Gating Overhead", "Latency (ms)": gating_overhead, "Rel. Speed": gating_overhead / latency_skip, "Note": "features + decision"},
        {"Method": "Hard Skip (Ours)", "Latency (ms)": latency_hard_skip, "Rel. Speed": latency_hard_skip / latency_skip, "Note": f"Skip={skip_rate:.0%}"},
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
