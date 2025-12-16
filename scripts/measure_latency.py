import time
import jax
import jax.numpy as jnp
from flax import nnx
from ponderttt.models import load_ttt_model


def benchmark():
    print("Setting up latency benchmark...")

    # Configuration
    # We use a small dummy config to verify relative speed, or load 125M if possible.
    # To be fast and robust, let's load a tiny proxy 125M (shallow) or just real 125M if it fits.
    # Let's try to load the real 125M structure but random weights to avoid downloading.
    # Actually, we can just use the config.

    # 125M Config (Approx)
    # GPT-2 small: 12 layers, 768 dim, 12 heads
    seq_len = 512
    batch_size = 1  # Latency per chunk usually implies BS=1 or small BS

    # Determine device
    print(f"JAX Devices: {jax.devices()}")

    print("Initializing 125M model...")
    # Use load_ttt_model but with random init to avoid weights
    # We can hack this by passing a config if allowed, or just trust it.
    # Let's use the real load if possible, or build manually.
    # load_ttt_model requires HF cache usually.
    # Let's try to just instantiate TTTModel directly with specific config.

    # Create a dummy config representing 125M
    # We don't have easy access to GPT2Config from here without transformers.
    # Let's try to grab the tokenizer and model via load_ttt_model, assuming it's cached or fast.
    # If it fails, I'll fallback.

    try:
        model, _ = load_ttt_model(
            "gpt2", fast_weight_type="ttt", load_pretrained=False
        )  # Random init
    except Exception as e:
        print(f"Failed to load standard model: {e}")
        return

    # JIT the step functions
    print("Compiling steps...")

    @nnx.jit(static_argnames=("use_ttt",))
    def forward_step(model, input_ids, use_ttt):
        # We need to simulate the 'generate' or 'forward' call.
        # The paper measures 'latency per 512-token chunk'. This usually means the processing time for one chunk.
        # For SKIP, it's just forward.
        # For UPDATE_1, it's forward + backward + update.
        # Our model() call handles internal TTT update if use_ttt=True.
        return model(input_ids, use_ttt=use_ttt)

    # Dummy Input
    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Helper to block on all leaves of a pytree
    def block_leaves(tree):
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    # Warmup
    print("Warmup SKIP (10 iters)...")
    for _ in range(10):
        out = forward_step(model, input_ids, use_ttt=False)
        block_leaves(out)

    print("Warmup UPDATE_1 (10 iters)...")
    for _ in range(10):
        out = forward_step(model, input_ids, use_ttt=True)
        block_leaves(out)

    print("Warmup complete")

    # Benchmark
    n_iters = 500

    # Measure SKIP
    block_leaves(input_ids)
    start = time.perf_counter()
    for _ in range(n_iters):
        out = forward_step(model, input_ids, use_ttt=False)
        block_leaves(out)
    end = time.perf_counter()
    skip_time_ms = (end - start) / n_iters * 1000
    print(f"SKIP Latency: {skip_time_ms:.2f} ms")

    # Measure UPDATE_1
    block_leaves(input_ids)
    start = time.perf_counter()
    for _ in range(n_iters):
        out = forward_step(model, input_ids, use_ttt=True)
        block_leaves(out)
    end = time.perf_counter()
    update_time_ms = (end - start) / n_iters * 1000
    print(f"UPDATE_1 Latency: {update_time_ms:.2f} ms")

    if skip_time_ms > 0:
        ratio = update_time_ms / skip_time_ms
        print(f"Ratio (UPDATE_1 / SKIP): {ratio:.2f}x")
    else:
        print("Error: SKIP latency is 0!")


if __name__ == "__main__":
    benchmark()
