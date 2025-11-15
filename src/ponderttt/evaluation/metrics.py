"""
Evaluation metrics for code generation and efficiency.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """
    Compute unbiased pass@k metric.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        pass@k score (0 to 1)

    Formula from "Evaluating Large Language Models Trained on Code" (Chen et al., 2021):
        pass@k = 1 - (n-c choose k) / (n choose k)
    """
    if n - c < k:
        return 1.0

    numerator = 1.0
    for i in range(k):
        numerator *= (n - c - i)

    denominator = 1.0
    for i in range(k):
        denominator *= (n - i)

    return 1.0 - (numerator / denominator)


def compute_flops(
    actions: List[str],
    base_flops: float = 1.0,
) -> float:
    """
    Compute total FLOPs based on actions taken.

    Args:
        actions: List of action names
        base_flops: Base FLOPs for a single forward pass

    Returns:
        Total FLOPs multiplier
    """
    action_costs = {
        "SKIP": 1.0,
        "UPDATE_1": 3.0,
        "UPDATE_2": 5.0,
        "UPDATE_4": 12.0,
    }

    total_cost = sum(action_costs.get(action, 1.0) for action in actions)
    return total_cost * base_flops


def compute_efficiency_metrics(
    quality_scores: List[float],
    costs: List[float],
) -> Dict[str, float]:
    """
    Compute efficiency metrics.

    Args:
        quality_scores: Quality scores (e.g., pass@k)
        costs: Computational costs (FLOPs multipliers)

    Returns:
        Dictionary with efficiency metrics
    """
    quality_scores = np.array(quality_scores)
    costs = np.array(costs)

    return {
        "mean_quality": float(np.mean(quality_scores)),
        "std_quality": float(np.std(quality_scores)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "quality_per_cost": float(np.mean(quality_scores / costs)),
        "efficiency_score": float(np.mean(quality_scores) / np.mean(costs)),
    }


def compute_pareto_frontier(
    methods: List[str],
    quality_scores: List[List[float]],
    costs: List[List[float]],
) -> Tuple[List[str], List[float], List[float]]:
    """
    Compute Pareto frontier of methods.

    Args:
        methods: Method names
        quality_scores: Quality scores for each method [method][sample]
        costs: Costs for each method [method][sample]

    Returns:
        Tuple of (pareto_methods, pareto_quality, pareto_costs)
    """
    # Compute mean quality and cost for each method
    mean_quality = [np.mean(q) for q in quality_scores]
    mean_cost = [np.mean(c) for c in costs]

    # Find Pareto-optimal points
    pareto_methods = []
    pareto_quality = []
    pareto_costs = []

    for i, (method, q, c) in enumerate(zip(methods, mean_quality, mean_cost)):
        is_pareto = True

        # Check if any other method dominates this one
        for j, (q2, c2) in enumerate(zip(mean_quality, mean_cost)):
            if i != j:
                # Method j dominates i if it has higher quality AND lower cost
                if q2 >= q and c2 <= c and (q2 > q or c2 < c):
                    is_pareto = False
                    break

        if is_pareto:
            pareto_methods.append(method)
            pareto_quality.append(q)
            pareto_costs.append(c)

    # Sort by cost
    sorted_indices = np.argsort(pareto_costs)
    pareto_methods = [pareto_methods[i] for i in sorted_indices]
    pareto_quality = [pareto_quality[i] for i in sorted_indices]
    pareto_costs = [pareto_costs[i] for i in sorted_indices]

    return pareto_methods, pareto_quality, pareto_costs


def compute_action_statistics(
    actions: List[str],
) -> Dict[str, float]:
    """
    Compute statistics about action distribution.

    Args:
        actions: List of actions taken

    Returns:
        Dictionary with action statistics
    """
    counter = Counter(actions)
    total = len(actions)

    stats = {
        f"freq_{action}": counter.get(action, 0) / total
        for action in ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    }

    # Compute entropy of action distribution
    probs = [counter.get(a, 0) / total for a in ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

    stats["entropy"] = float(entropy)
    stats["num_actions"] = total

    return stats


def compute_auc(
    x: List[float],
    y: List[float],
) -> float:
    """
    Compute Area Under Curve using trapezoidal rule.

    Args:
        x: X coordinates (should be sorted)
        y: Y coordinates

    Returns:
        AUC value
    """
    x = np.array(x)
    y = np.array(y)

    # Sort by x
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    # Trapezoidal rule
    auc = 0.0
    for i in range(len(x) - 1):
        width = x[i + 1] - x[i]
        height = (y[i] + y[i + 1]) / 2
        auc += width * height

    return float(auc)


def compute_oracle_agreement(
    predicted_actions: List[str],
    oracle_actions: List[str],
) -> float:
    """
    Compute agreement between predicted and oracle actions.

    Args:
        predicted_actions: Actions chosen by policy
        oracle_actions: Optimal actions (from oracle analysis)

    Returns:
        Agreement rate (0 to 1)
    """
    if len(predicted_actions) != len(oracle_actions):
        raise ValueError("Action lists must have same length")

    agreements = sum(p == o for p, o in zip(predicted_actions, oracle_actions))
    return agreements / len(predicted_actions)


def compute_correlation(
    x: List[float],
    y: List[float],
) -> Tuple[float, float]:
    """
    Compute Pearson and Spearman correlation.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Tuple of (pearson_r, spearman_rho)
    """
    x = np.array(x)
    y = np.array(y)

    # Pearson correlation
    pearson_r = float(np.corrcoef(x, y)[0, 1])

    # Spearman correlation (rank-based)
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))
    spearman_rho = float(np.corrcoef(x_ranks, y_ranks)[0, 1])

    return pearson_r, spearman_rho
