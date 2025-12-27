import time
import jax
import jax.numpy as jnp
from flax import nnx
from ponderttt.models import load_ttt_model
import threading
import subprocess
import shutil

# Try to import pynvml
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.utilization_samples = []
        self.memory_samples = []
        self.nvml_initialized = False
        self.thread = None

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except pynvml.NVMLError as e:
                print(
                    f"[GPUMonitor] pynvml init failed: {e}. Falling back to nvidia-smi."
                )
                self.nvml_initialized = False
        else:
            print("[GPUMonitor] pynvml not installed. Using nvidia-smi fallback.")

    def start(self):
        self.stop_event.clear()
        self.utilization_samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        # Do not shutdown NVML here to allow restart

    def shutdown(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            util, mem = self._get_stats()
            if util is not None:
                self.utilization_samples.append(util)
            if mem is not None:
                self.memory_samples.append(mem)
            time.sleep(self.interval)

    def _get_stats(self):
        if self.nvml_initialized:
            try:
                # Assume monitoring GPU 0 for simplicity, or aggregate
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # Utilization %, Memory Used (MB)
                return float(util_rates.gpu), float(mem_info.used) / 1024 / 1024
            except pynvml.NVMLError:
                return None, None
        else:
            # Fallback to nvidia-smi
            if not shutil.which("nvidia-smi"):
                return None, None

            try:
                # --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    return None, None

                # Output format: "gpu_util, memory_used" (e.g., "45, 1024")
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(",")
                return float(parts[0]), float(parts[1])
            except (subprocess.SubprocessError, ValueError, IndexError):
                return None, None

    def get_results(self):
        if not self.utilization_samples:
            return "N/A", "N/A"

        avg_util = sum(self.utilization_samples) / len(self.utilization_samples)
        max_mem = max(self.memory_samples) if self.memory_samples else 0.0
        return f"{avg_util:.1f}%", f"{max_mem:.1f} MB"


def benchmark(batch_size, model_scale="125m"):
    print("Setting up latency benchmark...")

    # Configuration
    seq_len = 512

    # Map scale to model name
    scale_to_model = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
        "xl": "gpt2-xl",
    }

    if model_scale not in scale_to_model:
        print(
            f"Invalid model scale: {model_scale}. Valid options: {list(scale_to_model.keys())}"
        )
        return

    model_name = scale_to_model[model_scale]

    # Determine device
    print(f"JAX Devices: {jax.devices()}")

    print(f"Initializing {model_scale.upper()} model ({model_name})...")
    try:
        model, _ = load_ttt_model(
            model_name, fast_weight_type="ttt", load_pretrained=False
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Compiling steps...")

    BLOCK_SIZE = 10  # 10 steps per block for 20% (2/10) and 50% (5/10) ratios

    @nnx.jit
    def forward_skip_block(model, input_ids):
        out = None
        for _ in range(BLOCK_SIZE):
            out = model(input_ids, use_ttt=False)
        return out

    @nnx.jit
    def forward_update_block(model, input_ids):
        out = None
        for _ in range(BLOCK_SIZE):
            out = model(input_ids, use_ttt=True)
        return out

    @nnx.jit
    def forward_sparse_20(model, input_ids):
        """20% update rate: S-S-S-S-U-S-S-S-S-U"""
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=True)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=False)
        out = model(input_ids, use_ttt=True)
        return out

    @nnx.jit
    def forward_periodic_50(model, input_ids):
        """Periodic 50% update rate (fixed schedule): S-U-S-U-S-U-S-U-S-U"""
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=True)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=True)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=True)
        _ = model(input_ids, use_ttt=False)
        _ = model(input_ids, use_ttt=True)
        _ = model(input_ids, use_ttt=False)
        out = model(input_ids, use_ttt=True)
        return out

    input_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    def block_leaves(tree):
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    # Warmup
    print("Warmup SKIP Block...")
    out = forward_skip_block(model, input_ids)
    block_leaves(out)

    print("Warmup UPDATE Block...")
    out = forward_update_block(model, input_ids)
    block_leaves(out)

    print("Warmup Ponder Sparse 20%...")
    out = forward_sparse_20(model, input_ids)
    block_leaves(out)

    print("Warmup Periodic 50%...")
    out = forward_periodic_50(model, input_ids)
    block_leaves(out)

    print("Warmup complete")

    # Init Monitor
    gpu_monitor = GPUMonitor(interval=0.05)

    # Run Benchmark
    # n_iters is total steps. We run n_iters // BLOCK_SIZE blocks.
    n_iters = 500
    n_blocks = n_iters // BLOCK_SIZE

    # 1. Measure SKIP (Fused)
    block_leaves(input_ids)
    print(f"Measuring SKIP ({n_iters} iters, blocked)...")
    gpu_monitor.start()
    start = time.perf_counter()
    for _ in range(n_blocks):
        out = forward_skip_block(model, input_ids)
        block_leaves(out)
    end = time.perf_counter()
    gpu_monitor.stop()

    skip_time_ms = (end - start) / (n_blocks * BLOCK_SIZE) * 1000
    skip_util, skip_mem = gpu_monitor.get_results()

    # 2. Measure UPDATE_1 (Fused)
    block_leaves(input_ids)
    print(f"Measuring UPDATE_1 ({n_iters} iters, blocked)...")
    gpu_monitor.start()
    start = time.perf_counter()
    for _ in range(n_blocks):
        out = forward_update_block(model, input_ids)
        block_leaves(out)
    end = time.perf_counter()
    gpu_monitor.stop()

    update_time_ms = (end - start) / (n_blocks * BLOCK_SIZE) * 1000
    update_util, update_mem = gpu_monitor.get_results()

    # 3. Measure PonderTTT [Dense] (Worst Case: 100% Update + Gating Overhead)
    block_leaves(input_ids)
    print(f"Measuring PonderTTT [Dense] ({n_iters} iters, blocked)...")
    gpu_monitor.start()
    start = time.perf_counter()
    for _ in range(n_blocks):
        out = forward_update_block(model, input_ids)
        block_leaves(out)

        # Emulate gating overhead: sync BLOCK_SIZE times (1 real + 9 dummy)
        if out and "ttt_stats" in out and out["ttt_stats"]:
            _ = float(jnp.mean(out["ttt_stats"]["ttt_loss_step_0"]))
            for _ in range(BLOCK_SIZE - 1):
                _ = float(0.0)
        else:
            for _ in range(BLOCK_SIZE):
                _ = float(0.0)
    end = time.perf_counter()
    gpu_monitor.stop()

    ponder_time_ms = (end - start) / (n_blocks * BLOCK_SIZE) * 1000
    ponder_util, ponder_mem = gpu_monitor.get_results()

    # 4. Measure PonderTTT [Sparse 20%]
    # Fused pattern
    print(f"Measuring PonderTTT [Sparse 20%] ({n_iters} iters, blocked)...")

    gpu_monitor.start()
    start = time.perf_counter()

    for _ in range(n_blocks):
        out = forward_sparse_20(model, input_ids)
        # Emulate gating overhead: sync BLOCK_SIZE times for consistency
        if out and "ttt_stats" in out and out["ttt_stats"]:
            _ = float(jnp.mean(out["ttt_stats"]["ttt_loss_step_0"]))
            for _ in range(BLOCK_SIZE - 1):
                _ = float(0.0)
        else:
            for _ in range(BLOCK_SIZE):
                _ = float(0.0)

    end = time.perf_counter()
    gpu_monitor.stop()

    # Total time covers n_blocks * 10 steps
    sparse_20_time_ms = (end - start) / (n_blocks * BLOCK_SIZE) * 1000
    sparse_20_util, sparse_20_mem = gpu_monitor.get_results()

    # 5. Measure PonderTTT [Sparse 50%]
    # Fused pattern
    print(f"Measuring Periodic 50% ({n_iters} iters, blocked)...")

    gpu_monitor.start()
    start = time.perf_counter()

    for _ in range(n_blocks):
        out = forward_periodic_50(model, input_ids)
        # Emulate gating overhead: sync BLOCK_SIZE times for consistency
        if out and "ttt_stats" in out and out["ttt_stats"]:
            _ = float(jnp.mean(out["ttt_stats"]["ttt_loss_step_0"]))
            for _ in range(BLOCK_SIZE - 1):
                _ = float(0.0)
        else:
            for _ in range(BLOCK_SIZE):
                _ = float(0.0)

    end = time.perf_counter()
    gpu_monitor.stop()

    # Total time covers n_blocks * 10 steps
    sparse_50_time_ms = (end - start) / (n_blocks * BLOCK_SIZE) * 1000
    sparse_50_util, sparse_50_mem = gpu_monitor.get_results()

    # Shutdown Monitor
    gpu_monitor.shutdown()

    # Build results dictionary for logging
    results = {
        "model_scale": model_scale,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_iters": n_iters,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": {
            "skip": round(skip_time_ms, 3),
            "update_1": round(update_time_ms, 3),
            "ponder_dense": round(ponder_time_ms, 3),
            "ponder_sparse_20": round(sparse_20_time_ms, 3),
            "ponder_sparse_50": round(sparse_50_time_ms, 3),
        },
        "relative_latency": {
            "skip": 1.0,
            "update_1": round(update_time_ms / skip_time_ms, 3)
            if skip_time_ms > 0
            else None,
            "ponder_dense": round(ponder_time_ms / skip_time_ms, 3)
            if skip_time_ms > 0
            else None,
            "ponder_sparse_20": round(sparse_20_time_ms / skip_time_ms, 3)
            if skip_time_ms > 0
            else None,
            "ponder_sparse_50": round(sparse_50_time_ms / skip_time_ms, 3)
            if skip_time_ms > 0
            else None,
        },
        "gpu_util": {
            "skip": skip_util,
            "update_1": update_util,
            "ponder_dense": ponder_util,
            "ponder_sparse_20": sparse_20_util,
            "ponder_sparse_50": sparse_50_util,
        },
        "max_vmem": {
            "skip": skip_mem,
            "update_1": update_mem,
            "ponder_dense": ponder_mem,
            "ponder_sparse_20": sparse_20_mem,
            "ponder_sparse_50": sparse_50_mem,
        },
    }

    # Results
    output_lines = []
    output_lines.append("\n" + "=" * 100)
    output_lines.append(
        f"Latency Benchmark Results (Model={model_scale.upper()}, Batch={batch_size}, N={n_iters})"
    )
    output_lines.append("=" * 100)
    output_lines.append(
        f"{'Metric':<15} | {'SKIP':<18} | {'UPDATE_1':<18} | {'Ponder[Dense]':<18} | {'Ponder[Sparse 20%]':<18} | {'Periodic 50%':<18}"
    )
    output_lines.append("-" * 100)
    output_lines.append(
        f"{'Latency (ms)':<15} | {skip_time_ms:<18.2f} | {update_time_ms:<18.2f} | {ponder_time_ms:<18.2f} | {sparse_20_time_ms:<18.2f} | {sparse_50_time_ms:<18.2f}"
    )

    if skip_time_ms > 0:
        r_upd = update_time_ms / skip_time_ms
        r_dense = ponder_time_ms / skip_time_ms
        r_sparse20 = sparse_20_time_ms / skip_time_ms
        r_sparse50 = sparse_50_time_ms / skip_time_ms
        output_lines.append(
            f"{'Acc. Factor':<15} | {'1.00x':<18} | {f'{r_upd:.2f}x':<18} | {f'{r_dense:.2f}x':<18} | {f'{r_sparse20:.2f}x':<18} | {f'{r_sparse50:.2f}x':<18}"
        )

    output_lines.append(
        f"{'Avg GPU Util':<15} | {skip_util:<18} | {update_util:<18} | {ponder_util:<18} | {sparse_20_util:<18} | {sparse_50_util:<18}"
    )
    output_lines.append(
        f"{'Max VMEM':<15} | {skip_mem:<18} | {update_mem:<18} | {ponder_mem:<18} | {sparse_20_mem:<18} | {sparse_50_mem:<18}"
    )
    output_lines.append("=" * 100 + "\n")
    output_lines.append(
        "Note: Ponder[Sparse 20%] simulates 20% update rate using fused kernel."
    )
    output_lines.append(
        "      Periodic 50% = fixed schedule baseline (every other chunk update)."
    )
    output_lines.append(
        "      Ponder[Dense] simulates 100% update rate + gating overhead (Worst Case)."
    )

    # Print to console
    for line in output_lines:
        print(line)

    # Save logs to file
    import json
    import os

    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)

    # Save JSON log
    json_path = os.path.join(log_dir, f"latency_{model_scale}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LOG] Results saved to: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for benchmark"
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="125m",
        choices=["125m", "350m", "1b", "xl"],
        help="Model scale to benchmark (default: 125m)",
    )
    args = parser.parse_args()
    benchmark(args.batch_size, args.model_scale)
