"""
Evaluation metrics and benchmarks for PonderTTT.
"""

from .benchmarks import (
    BenchmarkSuite,
    HumanEvalBenchmark,
    MBPPBenchmark,
)
from .metrics import (
    compute_action_statistics,
    compute_efficiency_metrics,
    compute_flops,
    compute_pareto_frontier,
    compute_pass_at_k,
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
