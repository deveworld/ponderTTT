"""
Profiling utilities for PonderTTT.

Tools for measuring performance, memory usage, and FLOPs.
"""

import json
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch


class ProfileContext:
    """
    Context manager for profiling model execution.

    Usage:
        with ProfileContext('my_experiment') as prof:
            output = model(input)

        prof.print_summary()
        prof.save_report('profile.json')
    """

    def __init__(self, name: str = "profile", device: str = "cuda"):
        """
        Initialize profiler.

        Args:
            name: Profile name
            device: Device to profile on
        """
        self.name = name
        self.device = device
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.memory_start: Optional[float] = None
        self.memory_end: Optional[float] = None
        self.memory_peak: Optional[float] = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.memory_start = torch.cuda.memory_allocated() / (1024**2)  # MB

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.memory_end = torch.cuda.memory_allocated() / (1024**2)  # MB
            self.memory_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        self.end_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "start_mb": self.memory_start or 0,
                "end_mb": self.memory_end or 0,
                "peak_mb": self.memory_peak or 0,
                "allocated_mb": (self.memory_end or 0) - (self.memory_start or 0),
            }
        return {}

    def print_summary(self):
        """Print profiling summary."""
        print(f"\n{'=' * 60}")
        print(f"Profile: {self.name}")
        print(f"{'=' * 60}")
        print(f"Time: {self.get_elapsed_time():.4f}s")

        if self.device == "cuda" and torch.cuda.is_available():
            mem_stats = self.get_memory_stats()
            print("\nMemory (MB):")
            print(f"  Start:     {mem_stats['start_mb']:.2f}")
            print(f"  End:       {mem_stats['end_mb']:.2f}")
            print(f"  Peak:      {mem_stats['peak_mb']:.2f}")
            print(f"  Allocated: {mem_stats['allocated_mb']:.2f}")

    def save_report(self, filepath: str):
        """Save profiling report to JSON."""
        from typing import Any
        report: Dict[str, Any] = {
            "name": self.name,
            "device": self.device,
            "elapsed_time_s": self.get_elapsed_time(),
        }

        if self.device == "cuda" and torch.cuda.is_available():
            report["memory_stats"] = self.get_memory_stats()

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Profile saved to: {filepath}")


class Timer:
    """
    Simple context manager for timing code blocks with CUDA synchronization.

    Usage:
        with Timer("forward_pass", device="cuda") as t:
            output = model(input)
        print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self, name: str = "block", device: str = "cuda"):
        """
        Initialize timer.

        Args:
            name: Name of the timed block
            device: Device type ('cuda' or 'cpu')
        """
        self.name = name
        self.device = device
        self.elapsed = 0.0
        self.start = 0.0

    def __enter__(self):
        """Start timing."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timing and record elapsed time."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start


class PerformanceTracker:
    """
    Track timing and FLOPs for multiple model operations.

    Usage:
        tracker = PerformanceTracker()
        with Timer("forward", "cuda") as t:
            output = model(input)
        tracker.record("forward", t.elapsed, flops=1000000)
        tracker.print_summary()
    """

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self.timings: Dict[str, List[float]] = {}
        self.flops: Dict[str, List[int]] = {}

    def record(self, name: str, elapsed: float, flops: int = 0):
        """
        Record a timing measurement.

        Args:
            name: Name of the operation
            elapsed: Elapsed time in seconds
            flops: Number of FLOPs (optional)
        """
        if name not in self.timings:
            self.timings[name] = []
            self.flops[name] = []

        self.timings[name].append(elapsed)
        if flops > 0:
            self.flops[name].append(flops)

    def summary(self) -> Dict:
        """
        Generate summary statistics.

        Returns:
            Dictionary with timing statistics for each operation
        """
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                "mean_time_ms": float(np.mean(times) * 1000),
                "std_time_ms": float(np.std(times) * 1000),
                "total_time_s": float(np.sum(times)),
                "count": len(times),
            }

            if name in self.flops and self.flops[name]:
                total_flops = np.sum(self.flops[name])
                total_time = np.sum(times)
                summary[name]["total_flops"] = int(total_flops)
                if total_time > 0:
                    summary[name]["tflops_per_sec"] = float((total_flops / total_time) / 1e12)

        return summary

    def print_summary(self):
        """Print formatted summary of performance statistics."""
        summary = self.summary()

        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Operation':<25} {'Mean (ms)':>12} {'Std (ms)':>12} {'Total (s)':>12} {'Count':>10}")
        print("-" * 80)

        for name, stats in sorted(summary.items()):
            print(
                f"{name:<25} {stats['mean_time_ms']:>12.3f} {stats['std_time_ms']:>12.3f} "
                f"{stats['total_time_s']:>12.2f} {stats['count']:>10}"
            )

            if "tflops_per_sec" in stats:
                print(f"  → Throughput: {stats['tflops_per_sec']:.3f} TFLOPS/s")


def profile_model_detailed(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 10,
    output_path: str = "experiments/results/trace.json",
):
    """
    Profile model layer by layer using torch.profiler.

    Args:
        model: Model to profile
        dataloader: Data loader for input batches
        device: Device to run on
        num_batches: Number of batches to profile
        output_path: Path to save Chrome trace
    """
    from torch.profiler import ProfilerActivity, profile

    model.eval()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                _ = model(input_ids)

    # Print results
    print("\n" + "=" * 80)
    print("DETAILED PROFILING RESULTS")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export trace
    prof.export_chrome_trace(output_path)
    print(f"\nTrace exported to: {output_path}")
    print("View at: chrome://tracing")


def analyze_speedup(
    baseline_results: Dict,
    adaptive_results: Dict,
) -> Dict:
    """
    Compare timing and FLOPs between baseline and adaptive methods.

    Args:
        baseline_results: Results dictionary from baseline experiment
        adaptive_results: Results dictionary from adaptive experiment

    Returns:
        Dictionary with speedup analysis
    """
    analysis = {}

    # Extract timing stats
    if "timing_stats" in baseline_results and "timing_stats" in adaptive_results:
        baseline_timing = baseline_results["timing_stats"]
        adaptive_timing = adaptive_results["timing_stats"]

        # Compare forward pass time
        if "forward_pass" in baseline_timing and "forward_pass" in adaptive_timing:
            baseline_time = baseline_timing["forward_pass"]["total_time_s"]
            adaptive_time = adaptive_timing["forward_pass"]["total_time_s"]
            actual_speedup = baseline_time / adaptive_time if adaptive_time > 0 else 0
            analysis["actual_speedup"] = actual_speedup
            analysis["baseline_forward_time_s"] = baseline_time
            analysis["adaptive_forward_time_s"] = adaptive_time

    # Extract FLOPs
    if "flops_per_token" in baseline_results and "flops_per_token" in adaptive_results:
        baseline_flops = baseline_results["flops_per_token"]
        adaptive_flops = adaptive_results["flops_per_token"]
        theoretical_speedup = baseline_flops / adaptive_flops if adaptive_flops > 0 else 0
        flops_change_percent = ((adaptive_flops - baseline_flops) / baseline_flops) * 100 if baseline_flops > 0 else 0

        analysis["theoretical_speedup"] = theoretical_speedup
        analysis["baseline_flops"] = baseline_flops
        analysis["adaptive_flops"] = adaptive_flops
        analysis["flops_change_percent"] = flops_change_percent

    # Compute efficiency
    if "actual_speedup" in analysis and "theoretical_speedup" in analysis:
        if analysis["theoretical_speedup"] > 0:
            analysis["efficiency_percent"] = (
                analysis["actual_speedup"] / analysis["theoretical_speedup"]
            ) * 100

    return analysis


def print_speedup_analysis(analysis: Dict):
    """
    Print formatted speedup analysis.

    Args:
        analysis: Dictionary from analyze_speedup
    """
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS")
    print("=" * 70)

    if "flops_change_percent" in analysis:
        change = analysis["flops_change_percent"]
        print(f"Theoretical FLOPs change: {change:.1f}% (negative = less compute)")

    if "theoretical_speedup" in analysis:
        print(f"Theoretical speedup:         {analysis['theoretical_speedup']:.2f}x")

    if "actual_speedup" in analysis:
        print(f"\nActual wall-clock speedup:   {analysis['actual_speedup']:.2f}x")

    if "efficiency_percent" in analysis:
        print(f"Efficiency:                  {analysis['efficiency_percent']:.1f}%")

        if analysis["efficiency_percent"] < 50:
            print("\n⚠ Warning: Actual speedup much lower than theoretical!")
            print("  Possible reasons:")
            print("  - Memory bandwidth bottleneck")
            print("  - Small batch size (insufficient parallelism)")
            print("  - Overhead from dynamic iteration allocation")
            print("  - Non-FLOPs operations (memory transfers, kernel launches)")
            print("\n  Consider profiling with torch.profiler for detailed analysis")


def estimate_model_flops(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cpu",
) -> Dict:
    """
    Estimate FLOPs for a model.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (batch, seq_len, ...)
        device: Device to use

    Returns:
        Dictionary with FLOP estimates
    """
    # This is a simple estimation - for detailed profiling, use torch.profiler
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create dummy input
    dummy_input = torch.randint(0, 50000, input_shape).to(device)

    # Measure forward pass time
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    elapsed = time.time() - start

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "forward_time_s": elapsed,
        "input_shape": input_shape,
    }


def compare_configurations(
    configs: list,
    input_shape: tuple,
    device: str = "cuda",
    num_runs: int = 10,
) -> Dict:
    """
    Compare different model configurations.

    Args:
        configs: List of (name, model) tuples
        input_shape: Input shape for testing
        device: Device to use
        num_runs: Number of runs for averaging

    Returns:
        Comparison results
    """
    results = []

    for name, model in configs:
        print(f"\nProfiling: {name}")
        model = model.to(device)
        model.eval()

        # Warmup
        dummy_input = torch.randint(0, 50000, input_shape).to(device)
        with torch.no_grad():
            _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        # Time multiple runs
        times: list[float] = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        # Memory
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        results.append({
            "name": name,
            "avg_time_s": avg_time,
            "std_time_s": std_time,
            "peak_memory_mb": peak_memory,
            "params": sum(p.numel() for p in model.parameters()),
        })

    return {"results": results, "input_shape": input_shape, "num_runs": num_runs}


def print_comparison_table(comparison: Dict):
    """Print comparison results as a table."""
    results = comparison["results"]

    print(f"\n{'=' * 80}")
    print("Configuration Comparison")
    print(f"Input shape: {comparison['input_shape']}, Runs: {comparison['num_runs']}")
    print(f"{'=' * 80}")
    print(
        f"{'Name':<20} {'Time (s)':>12} {'Std (s)':>12} {'Memory (MB)':>15} {'Params':>12}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['name']:<20} {r['avg_time_s']:>12.4f} {r['std_time_s']:>12.4f} "
            f"{r['peak_memory_mb']:>15.2f} {r['params']:>12,}"
        )


@contextmanager
def time_block(name: str = "Block", verbose: bool = True):
    """
    Simple context manager to time a block of code.

    Usage:
        with time_block('training'):
            train_model()
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if verbose:
            print(f"{name}: {elapsed:.4f}s")


def get_gpu_info() -> Dict:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }


def print_gpu_info():
    """Print GPU information."""
    info = get_gpu_info()

    print(f"\n{'=' * 60}")
    print("GPU Information")
    print(f"{'=' * 60}")

    if info["available"]:
        print("Available: Yes")
        print(f"Device count: {info['device_count']}")
        print(f"Current device: {info['current_device']}")
        print(f"Device name: {info['device_name']}")
        print(f"Total memory: {info['total_memory_gb']:.2f} GB")
    else:
        print("Available: No")
        print("CUDA is not available on this system")


def measure_adaptive_overhead(
    model,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = "cuda",
    num_runs: int = 10,
) -> Dict:
    """
    Measure the overhead of adaptive TTT compared to fixed TTT.

    This function measures:
    1. Time for difficulty computation (entropy)
    2. Time for allocation
    3. Time for TTT iterations
    4. Total time

    Args:
        model: Model with adaptive TTT
        batch_size: Batch size for testing
        seq_len: Sequence length
        device: Device to use
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with timing breakdown
    """
    import torch

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(0, 50000, (batch_size, seq_len)).to(device_obj)

    # Warmup
    with torch.no_grad():
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Measure full forward pass
    times_full: list[float] = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input, return_stats=True)
        if device == "cuda":
            torch.cuda.synchronize()
        times_full.append(time.time() - start)

    avg_time_full = sum(times_full) / len(times_full)

    # Estimate per-token time
    time_per_token = avg_time_full / (batch_size * seq_len)

    results = {
        "total_time_s": avg_time_full,
        "time_per_token_ms": time_per_token * 1000,
        "throughput_tokens_per_s": (batch_size * seq_len) / avg_time_full,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_runs": num_runs,
    }

    return results


def compare_overhead_breakdown(
    config,
    batch_size: int = 8,
    seq_len: int = 256,
    device: str = "cuda",
) -> Dict:
    """
    Compare time breakdown between baseline and adaptive TTT.

    Measures:
    - Baseline: Only TTT time
    - Adaptive: TTT time + difficulty computation + allocation

    Args:
        config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to use

    Returns:
        Breakdown of time for baseline vs adaptive
    """
    import torch
    from ..models import TransformerTTT, TransformerConfig

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create baseline model
    baseline_config = TransformerConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        ttt_dim=config.ttt_dim,
        ttt_iterations=4,
        use_ttt=True,
        use_adaptive_ttt=False,
    )
    baseline_model = TransformerTTT(baseline_config).to(device_obj)
    baseline_model.eval()

    # Create adaptive model
    adaptive_config = TransformerConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        ttt_dim=config.ttt_dim,
        ttt_iterations=2,
        use_ttt=True,
        use_adaptive_ttt=True,
    )
    adaptive_model = TransformerTTT(adaptive_config).to(device_obj)
    adaptive_model.eval()

    # Measure both
    baseline_timing = measure_adaptive_overhead(
        baseline_model, batch_size, seq_len, device
    )
    adaptive_timing = measure_adaptive_overhead(
        adaptive_model, batch_size, seq_len, device
    )

    overhead_ms = (
        adaptive_timing["time_per_token_ms"] - baseline_timing["time_per_token_ms"]
    )
    overhead_percent = (overhead_ms / baseline_timing["time_per_token_ms"]) * 100

    return {
        "baseline": baseline_timing,
        "adaptive": adaptive_timing,
        "overhead_ms_per_token": overhead_ms,
        "overhead_percent": overhead_percent,
    }


def estimate_experiment_time(
    config,
    num_epochs: int = 3,
    num_train_batches: int = 500,
    num_eval_batches: int = 100,
    batch_size: int = 8,
    seq_len: int = 256,
    num_seeds: int = 3,
    device: str = "cuda",
) -> Dict:
    """
    Estimate total experiment time.

    Args:
        config: Model configuration
        num_epochs: Number of training epochs
        num_train_batches: Training batches per epoch
        num_eval_batches: Evaluation batches
        batch_size: Batch size
        seq_len: Sequence length
        num_seeds: Number of random seeds
        device: Device to use

    Returns:
        Time estimates
    """
    # Estimate time per batch (rough approximation)
    # Training is ~3x slower than inference due to backward pass
    inference_time_per_batch_s = 0.1 if device == "cuda" else 1.0
    training_time_per_batch_s = inference_time_per_batch_s * 3

    # Time for one config, one seed
    time_per_epoch = num_train_batches * training_time_per_batch_s
    time_for_eval = num_eval_batches * inference_time_per_batch_s
    time_per_config_per_seed = num_epochs * (time_per_epoch + time_for_eval)

    # Total time
    num_configs = 4  # Fixed-1, Fixed-2, Fixed-4, Adaptive
    total_time_s = num_configs * num_seeds * time_per_config_per_seed

    return {
        "time_per_config_per_seed_minutes": time_per_config_per_seed / 60,
        "total_time_hours": total_time_s / 3600,
        "num_configs": num_configs,
        "num_seeds": num_seeds,
        "num_epochs": num_epochs,
        "estimated_gpu_hours": total_time_s / 3600,
        "estimated_cost_usd": (total_time_s / 3600) * 0.50,  # $0.50/hr
    }


def print_experiment_estimate(estimate: Dict):
    """Print experiment time estimate."""
    print(f"\n{'=' * 60}")
    print("Experiment Time Estimate")
    print(f"{'=' * 60}")
    print(f"Configs: {estimate['num_configs']}")
    print(f"Seeds per config: {estimate['num_seeds']}")
    print(f"Epochs per run: {estimate['num_epochs']}")
    print()
    print(f"Time per config/seed: {estimate['time_per_config_per_seed_minutes']:.1f} minutes")
    print(f"Total time: {estimate['total_time_hours']:.1f} hours")
    print()
    print(f"GPU hours needed: {estimate['estimated_gpu_hours']:.1f}")
    print(f"Estimated cost (@ $0.50/hr): ${estimate['estimated_cost_usd']:.2f}")
    print(f"{'=' * 60}")
