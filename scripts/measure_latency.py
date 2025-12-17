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
        self.thread = threading.Thread(target=self._monitor_loop)
        self.nvml_initialized = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                print("[GPUMonitor] Using pynvml for GPU monitoring.")
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
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
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


def benchmark():
    print("Setting up latency benchmark...")

    # Configuration
    seq_len = 512
    batch_size = 1

    # Determine device
    print(f"JAX Devices: {jax.devices()}")

    print("Initializing 125M model...")
    try:
        model, _ = load_ttt_model("gpt2", fast_weight_type="ttt", load_pretrained=False)
    except Exception as e:
        print(f"Failed to load standard model: {e}")
        return

    # JIT the step functions
    print("Compiling steps...")

    @nnx.jit(static_argnames=("use_ttt",))
    def forward_step(model, input_ids, use_ttt):
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

    # GPU Monitor
    gpu_monitor = GPUMonitor(interval=0.05)
    gpu_monitor.start()

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

    # Measure UPDATE_1
    block_leaves(input_ids)
    start = time.perf_counter()
    for _ in range(n_iters):
        out = forward_step(model, input_ids, use_ttt=True)
        block_leaves(out)
    end = time.perf_counter()
    update_time_ms = (end - start) / n_iters * 1000

    # Stop Monitor
    gpu_monitor.stop()
    avg_util, max_mem = gpu_monitor.get_results()

    # Results
    print("\n" + "=" * 40)
    print(f"Latency Benchmark Results (N={n_iters})")
    print("=" * 40)
    print(f"SKIP Latency:      {skip_time_ms:.2f} ms")
    print(f"UPDATE_1 Latency:  {update_time_ms:.2f} ms")

    if skip_time_ms > 0:
        ratio = update_time_ms / skip_time_ms
        print(f"Ratio (UPD/SKIP):  {ratio:.2f}x")
    else:
        print("Error: SKIP latency is 0!")

    print(f"Avg GPU Util:      {avg_util}")
    print(f"Max VMEM Used:     {max_mem}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    benchmark()
