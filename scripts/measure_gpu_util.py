"""
Measure GPU Utilization for PonderTTT Methods.

This script measures GPU utilization (not wall-clock latency) for different
TTT gating strategies. Due to XLA kernel fusion, wall-clock latency shows
minimal difference between SKIP and UPDATE. GPU utilization is a more
meaningful metric showing the actual compute workload.

Usage:
    python scripts/measure_gpu_util.py --model_scale 125m --batch_size 1
"""

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
    """Monitor GPU utilization during benchmark."""

    def __init__(self, interval=0.05):
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
                print(f"[GPUMonitor] pynvml init failed: {e}.")
                self.nvml_initialized = False

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
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return float(util_rates.gpu), float(mem_info.used) / 1024 / 1024
            except pynvml.NVMLError:
                return None, None
        else:
            if not shutil.which("nvidia-smi"):
                return None, None
            try:
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
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(",")
                return float(parts[0]), float(parts[1])
            except (subprocess.SubprocessError, ValueError, IndexError):
                return None, None

    def get_results(self):
        if not self.utilization_samples:
            return None, None
        avg_util = sum(self.utilization_samples) / len(self.utilization_samples)
        max_mem = max(self.memory_samples) if self.memory_samples else 0.0
        return avg_util, max_mem


def benchmark(batch_size, model_scale="125m"):
    print("Setting up GPU utilization benchmark...")

    seq_len = 512
    n_iters = 300  # Reduced for faster measurement

    scale_to_model = {
        "125m": "gpt2",
        "350m": "gpt2-medium",
        "1b": "gpt2-large",
        "xl": "gpt2-xl",
    }

    if model_scale not in scale_to_model:
        print(f"Invalid model scale: {model_scale}.")
        return

    model_name = scale_to_model[model_scale]

    devices = jax.devices()
    print(f"JAX Devices: {devices}")

    # Get GPU name
    gpu_name = "Unknown"
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            pynvml.nvmlShutdown()
        except Exception:
            pass

    print(f"Initializing {model_scale.upper()} model ({model_name})...")
    model, _ = load_ttt_model(model_name, fast_weight_type="ttt", load_pretrained=False)

    # JIT compile single-step functions
    @nnx.jit
    def forward_skip(model, input_ids):
        return model(input_ids, use_ttt=False)

    @nnx.jit
    def forward_update(model, input_ids):
        return model(input_ids, use_ttt=True)

    def block_leaves(tree):
        jax.tree.map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            tree,
        )

    # Generate different inputs
    print("Generating random input sequences...")
    rng = jax.random.PRNGKey(42)
    all_input_ids = []
    for i in range(n_iters):
        rng, subkey = jax.random.split(rng)
        input_ids = jax.random.randint(subkey, (batch_size, seq_len), 0, 50257)
        all_input_ids.append(jax.device_put(input_ids))

    # Warmup
    print("Warmup...")
    for i in range(10):
        out = forward_skip(model, all_input_ids[i])
        block_leaves(out)
        out = forward_update(model, all_input_ids[i])
        block_leaves(out)
    print("Warmup complete")

    gpu_monitor = GPUMonitor(interval=0.02)

    results = {}

    # === 1. SKIP ===
    print(f"Measuring SKIP (n={n_iters})...")
    gpu_monitor.start()
    for i in range(n_iters):
        out = forward_skip(model, all_input_ids[i])
        block_leaves(out)
    gpu_monitor.stop()
    skip_util, skip_mem = gpu_monitor.get_results()
    results["SKIP"] = {"util": skip_util, "mem": skip_mem, "flops": 1.0}

    # === 2. UPDATE_1 ===
    print(f"Measuring UPDATE_1 (n={n_iters})...")
    gpu_monitor.start()
    for i in range(n_iters):
        out = forward_update(model, all_input_ids[i])
        block_leaves(out)
    gpu_monitor.stop()
    update_util, update_mem = gpu_monitor.get_results()
    results["UPDATE_1"] = {"util": update_util, "mem": update_mem, "flops": 3.0}

    # === 3. Periodic 50% ===
    print(f"Measuring Periodic 50% (n={n_iters})...")
    gpu_monitor.start()
    for i in range(n_iters):
        if i % 2 == 0:
            out = forward_skip(model, all_input_ids[i])
        else:
            out = forward_update(model, all_input_ids[i])
        block_leaves(out)
    gpu_monitor.stop()
    p50_util, p50_mem = gpu_monitor.get_results()
    results["PonderTTT_50"] = {"util": p50_util, "mem": p50_mem, "flops": 2.0}

    gpu_monitor.shutdown()

    # Print results
    print("\n" + "=" * 80)
    print(
        f"GPU Utilization Benchmark (Model={model_scale.upper()}, Batch={batch_size})"
    )
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print("-" * 80)
    print(
        f"{'Method':<20} | {'Rel. FLOPs':<12} | {'GPU Util (%)':<15} | {'Max VMEM (MB)':<15}"
    )
    print("-" * 80)

    for method, data in results.items():
        util_str = f"{data['util']:.1f}" if data["util"] else "N/A"
        mem_str = f"{data['mem']:.0f}" if data["mem"] else "N/A"
        print(
            f"{method:<20} | {data['flops']:.1f}×{'':<9} | {util_str:<15} | {mem_str:<15}"
        )

    print("=" * 80)
    print("\nNote: GPU utilization shows actual compute workload.")
    print("      Wall-clock latency may not differ due to XLA kernel fusion.")

    # Save JSON
    import json
    import os

    log_dir = "outputs"
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"gpu_util_{model_scale}.json")

    output = {
        "model_scale": model_scale,
        "model_name": model_name,
        "gpu_name": gpu_name,
        "batch_size": batch_size,
        "n_iters": n_iters,
        "results": {
            k: {
                "gpu_util_pct": v["util"],
                "max_vmem_mb": v["mem"],
                "rel_flops": v["flops"],
            }
            for k, v in results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[LOG] Saved to: {json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--model_scale", type=str, default="125m", choices=["125m", "350m", "1b", "xl"]
    )
    args = parser.parse_args()
    benchmark(args.batch_size, args.model_scale)
