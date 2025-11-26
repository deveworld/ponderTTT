"""
Experiment C: Wall-clock Latency Measurement
Measures end-to-end inference time (ms/token) for different methods.
"""

import time
import jax
import jax.numpy as jnp
from flax import nnx
import pandas as pd

from ponderttt.models import load_ttt_model
from ponderttt.models.gating_nnx import GatingConfig, GatingNetwork

def benchmark_latency(model_scale="125m", chunk_size=512, num_runs=50):
    print(f"Benchmarking Latency: {model_scale}, Chunk={chunk_size}")
    
    # 1. Setup
    model_name = {"125m": "gpt2", "350m": "gpt2-medium", "1b": "gpt2-large"}[model_scale]
    
    # Load Model
    # random weights ok for timing
    model, _ = load_ttt_model(model_name=model_name, fast_weight_type="ttt", load_pretrained=False)
    
    # Gating Network
    gating_net = GatingNetwork(
        config=GatingConfig(feature_dim=32, hidden_dim=64), 
        rngs=nnx.Rngs(0)
    )
    
    # Dummy Inputs
    batch_size = 1
    input_ids = jnp.ones((batch_size, chunk_size), dtype=jnp.int32)
    
    # JIT Compilation
    print("Compiling functions...")
    
    @nnx.jit
    def step_skip(model, x):
        return model(x, use_ttt=False)
        
    @nnx.jit
    def step_update1(model, x):
        # Simulate 1 step update
        scale = jnp.array([[1.0]])
        return model(x, use_ttt=True, gating_scale=scale)

    @nnx.jit
    def run_gate(net, x): 
        return net(x, train=False)

    # Warmup
    print("Warmup...")
    step_skip(model, input_ids)
    step_update1(model, input_ids)
    feats = jnp.ones((1, 32))
    run_gate(gating_net, feats)
    
    # Measure SKIP
    jax.block_until_ready(step_skip(model, input_ids))
    start = time.time()
    for _ in range(num_runs):
        out = step_skip(model, input_ids)
        jax.block_until_ready(out)
    t_skip = (time.time() - start) / num_runs * 1000
    
    # Measure UPDATE_1
    jax.block_until_ready(step_update1(model, input_ids))
    start = time.time()
    for _ in range(num_runs):
        out = step_update1(model, input_ids)
        jax.block_until_ready(out)
    t_update1 = (time.time() - start) / num_runs * 1000
    
    # Measure Gating Net Cost
    jax.block_until_ready(run_gate(gating_net, feats))
    start = time.time()
    for _ in range(num_runs):
        o = run_gate(gating_net, feats)
        jax.block_until_ready(o)
    t_gate = (time.time() - start) / num_runs * 1000
    
    # Calculate Expected Latency for Diff (Budget 1.5 -> 0.18 steps)
    avg_steps = 0.18
    overhead = t_update1 - t_skip # Cost of TTT layer
    t_diff = t_skip + t_gate + (overhead * avg_steps)
    
    results = [
        {"Method": "SKIP (Baseline)", "Latency (ms)": t_skip, "Rel. Speed": 1.0},
        {"Method": "UPDATE_1 (Fixed)", "Latency (ms)": t_update1, "Rel. Speed": t_update1/t_skip},
        {"Method": "Gating Overhead", "Latency (ms)": t_gate, "Rel. Speed": t_gate/t_skip},
        {"Method": "Diff Gating (Avg)", "Latency (ms)": t_diff, "Rel. Speed": t_diff/t_skip, "Note": f"AvgSteps={avg_steps}"}
    ]
    
    df = pd.DataFrame(results)
    print("\n=== Latency Results (ms per chunk) ===")
    print(df)
    
    # Print speedup analysis
    print(f"\nSpeedup vs UPDATE_1: {t_update1 / t_diff:.2f}x")

if __name__ == "__main__":
    benchmark_latency()
