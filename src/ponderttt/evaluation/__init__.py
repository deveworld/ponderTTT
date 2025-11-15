"""
Evaluation metrics and benchmarks for PonderTTT.
"""

from .metrics import (
    compute_pass_at_k,
    compute_flops,
    compute_efficiency_metrics,
    compute_pareto_frontier,
    compute_action_statistics,
)
from .benchmarks import (
    HumanEvalBenchmark,
    MBPPBenchmark,
    BenchmarkSuite,
)

__all__ = [
    # Metrics
    "compute_pass_at_k",
    "compute_flops",
    "compute_efficiency_metrics",
    "compute_pareto_frontier",
    "compute_action_statistics",
    # Benchmarks
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "BenchmarkSuite",
]
