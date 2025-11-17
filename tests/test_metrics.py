"""
Unit tests for evaluation metrics.
"""

import pytest

from ponderttt.evaluation.metrics import (
    compute_pass_at_k,
    compute_flops,
    compute_efficiency_metrics,
    compute_pareto_frontier,
)


class TestPassAtK:
    """Test suite for pass@k metric."""

    def test_perfect_performance(self):
        """Test pass@k with all samples correct."""
        n = 10
        c = 10
        k = 1

        result = compute_pass_at_k(n, c, k)
        assert result == 1.0

    def test_zero_performance(self):
        """Test pass@k with no correct samples."""
        n = 10
        c = 0
        k = 1

        result = compute_pass_at_k(n, c, k)
        assert result == 0.0

    def test_partial_performance(self):
        """Test pass@k with some correct samples."""
        n = 10
        c = 5
        k = 1

        result = compute_pass_at_k(n, c, k)
        assert 0.0 < result < 1.0
        assert result == pytest.approx(0.5, rel=0.1)

    def test_k_greater_than_n(self):
        """Test pass@k with k > n (edge case)."""
        # Implementation handles this gracefully
        result = compute_pass_at_k(n=5, c=3, k=10)
        # Should return a value (implementation specific)
        assert isinstance(result, float)

    def test_c_greater_than_n(self):
        """Test pass@k with c > n (edge case)."""
        # Implementation handles this gracefully
        result = compute_pass_at_k(n=5, c=10, k=1)
        # Should return a value (implementation specific)
        assert isinstance(result, float)


class TestFLOPs:
    """Test suite for FLOP computation."""

    def test_skip_only(self):
        """Test FLOPs with only SKIP actions."""
        actions = ['SKIP', 'SKIP', 'SKIP']
        flops = compute_flops(actions, base_flops=1.0)
        # SKIP has cost 1.0, so 3 SKIPs = 3.0
        assert flops == 3.0

    def test_mixed_actions(self):
        """Test FLOPs with mixed actions."""
        actions = ['SKIP', 'UPDATE_1', 'UPDATE_2']
        flops = compute_flops(actions, base_flops=1.0)
        # SKIP=1.0, UPDATE_1=3.0, UPDATE_2=6.0, total=10.0
        assert flops == 10.0

    def test_update_4_action(self):
        """Test FLOPs with UPDATE_4 action."""
        actions = ['UPDATE_4']
        flops = compute_flops(actions, base_flops=1.0)
        # UPDATE_4 has cost 12.0
        assert flops == 12.0

    def test_base_flops_scaling(self):
        """Test that base_flops scales the result."""
        actions = ['SKIP', 'UPDATE_1']
        flops1 = compute_flops(actions, base_flops=1.0)
        flops2 = compute_flops(actions, base_flops=2.0)

        # Should scale linearly
        assert flops2 == flops1 * 2.0


class TestEfficiencyMetrics:
    """Test suite for efficiency metrics."""

    def test_efficiency_metrics_computation(self):
        """Test basic efficiency metrics computation."""
        quality_scores = [0.8, 0.7, 0.9]
        costs = [5.0, 4.0, 6.0]

        metrics = compute_efficiency_metrics(quality_scores, costs)

        # Should return a dictionary
        assert isinstance(metrics, dict)
        assert 'mean_quality' in metrics
        assert 'mean_cost' in metrics
        assert 'efficiency_score' in metrics

    def test_efficiency_metrics_values(self):
        """Test that efficiency metrics contain correct values."""
        quality_scores = [0.8, 0.8, 0.8]
        costs = [5.0, 5.0, 5.0]

        metrics = compute_efficiency_metrics(quality_scores, costs)

        assert metrics['mean_quality'] == pytest.approx(0.8)
        assert metrics['mean_cost'] == pytest.approx(5.0)
        # Efficiency should be quality/cost
        assert metrics['efficiency_score'] == pytest.approx(0.8 / 5.0)


class TestParetoFrontier:
    """Test suite for Pareto frontier computation."""

    def test_single_point(self):
        """Test Pareto frontier with single point."""
        methods = ['method1']
        qualities = [[0.8, 0.8, 0.8]]  # list of lists
        costs = [[5.0, 5.0, 5.0]]

        frontier_methods, frontier_qualities, frontier_costs = compute_pareto_frontier(
            methods, qualities, costs
        )

        assert len(frontier_methods) == 1
        assert frontier_methods[0] == 'method1'

    def test_dominated_points(self):
        """Test that dominated points are excluded."""
        methods = ['worse', 'better']
        qualities = [[0.6, 0.6], [0.8, 0.8]]  # list of lists
        costs = [[10.0, 10.0], [5.0, 5.0]]

        frontier_methods, frontier_qualities, frontier_costs = compute_pareto_frontier(
            methods, qualities, costs
        )

        # Only 'better' should be on frontier (higher quality, lower cost)
        assert 'better' in frontier_methods
        assert 'worse' not in frontier_methods

    def test_multiple_frontier_points(self):
        """Test frontier with multiple non-dominated points."""
        methods = ['fast', 'balanced', 'accurate']
        qualities = [[0.6, 0.6], [0.7, 0.7], [0.9, 0.9]]
        costs = [[2.0, 2.0], [5.0, 5.0], [10.0, 10.0]]

        frontier_methods, frontier_qualities, frontier_costs = compute_pareto_frontier(
            methods, qualities, costs
        )

        # All should be on frontier (trade-off between quality and cost)
        assert len(frontier_methods) == 3

    def test_frontier_sorted(self):
        """Test that frontier is sorted by cost."""
        methods = ['expensive', 'cheap', 'medium']
        qualities = [[0.9, 0.9], [0.6, 0.6], [0.7, 0.7]]
        costs = [[10.0, 10.0], [2.0, 2.0], [5.0, 5.0]]

        frontier_methods, frontier_qualities, frontier_costs = compute_pareto_frontier(
            methods, qualities, costs
        )

        # Should be sorted by increasing cost
        for i in range(len(frontier_costs) - 1):
            assert frontier_costs[i] <= frontier_costs[i + 1]
