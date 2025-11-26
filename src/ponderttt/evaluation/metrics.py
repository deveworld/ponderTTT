"""
Evaluation metrics for code generation and efficiency.
"""

from collections import Counter

import numpy as np


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
    # Edge cases
    if k <= 0 or n <= 0 or c <= 0:
        return 0.0
    if k > n:
        k = n

    # 1 - prod_{i=0..k-1} (n-c-i) / (n-i)
    numerator = 1.0
    denominator = 1.0
    for i in range(k):
        numerator *= max(n - c - i, 0)
        denominator *= n - i

    if denominator <= 0:
        return 0.0

    ratio = numerator / denominator
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def compute_flops(
    actions: list[str],
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
    # Cost model: 1 (base forward) + 2 * num_steps
    action_costs = {
        "SKIP": 1.0,      # 1 + 2*0 = 1
        "UPDATE_1": 3.0,  # 1 + 2*1 = 3
        "UPDATE_2": 5.0,  # 1 + 2*2 = 5
        "UPDATE_4": 9.0,  # 1 + 2*4 = 9
    }

    total_cost = sum(action_costs.get(action, 1.0) for action in actions)
    return total_cost * base_flops


def compute_efficiency_metrics(
    quality_scores: list[float],
    costs: list[float],
) -> dict[str, float]:
    """
    Compute efficiency metrics.

    Args:
        quality_scores: Quality scores (e.g., pass@k)
        costs: Computational costs (FLOPs multipliers)

    Returns:
        Dictionary with efficiency metrics
    """
    quality_scores_np = np.array(quality_scores, dtype=float)
    costs_np = np.array(costs, dtype=float)

    # Prevent division by zero/NaN in efficiency calculations
    safe_costs = np.where(costs_np == 0, np.inf, costs_np)

    return {
        "mean_quality": float(np.mean(quality_scores_np)),
        "std_quality": float(np.std(quality_scores_np)),
        "mean_cost": float(np.mean(costs_np)),
        "std_cost": float(np.std(costs_np)),
        "quality_per_cost": float(np.mean(quality_scores_np / safe_costs)),
        "efficiency_score": float(np.mean(quality_scores_np) / np.mean(safe_costs)),
    }


def compute_pareto_frontier(
    methods: list[str],
    quality_scores: list[list[float]],
    costs: list[list[float]],
) -> tuple[list[str], list[float], list[float]]:
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
            pareto_quality.append(float(q))
            pareto_costs.append(float(c))

    # Sort by cost
    sorted_indices = np.argsort(pareto_costs)
    pareto_methods = [pareto_methods[i] for i in sorted_indices]
    pareto_quality = [pareto_quality[i] for i in sorted_indices]
    pareto_costs = [pareto_costs[i] for i in sorted_indices]

    return pareto_methods, pareto_quality, pareto_costs


def compute_action_statistics(
    actions: list[str],
) -> dict[str, float]:
    """
    Compute statistics about action distribution.

    Args:
        actions: List of actions taken

    Returns:
        Dictionary with action statistics
    """
    if not actions:
        return {
            "freq_SKIP": 0.0,
            "freq_UPDATE_1": 0.0,
            "freq_UPDATE_2": 0.0,
            "freq_UPDATE_4": 0.0,
            "entropy": 0.0,
            "num_actions": 0,
        }

    counter = Counter(actions)
    total = len(actions)

    stats = {
        f"freq_{action}": counter.get(action, 0) / total
        for action in ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    }

    # Compute entropy of action distribution
    probs = [
        counter.get(a, 0) / total for a in ["SKIP", "UPDATE_1", "UPDATE_2", "UPDATE_4"]
    ]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

    stats["entropy"] = float(entropy)
    stats["num_actions"] = total

    return stats


def compute_auc(
    x: list[float],
    y: list[float],
) -> float:
    """
    Compute Area Under Curve using trapezoidal rule.

    Args:
        x: X coordinates (should be sorted)
        y: Y coordinates

    Returns:
        AUC value
    """
    x_np = np.array(x)
    y_np = np.array(y)

    # Sort by x
    sorted_indices = np.argsort(x_np)
    x_sorted = x_np[sorted_indices]
    y_sorted = y_np[sorted_indices]

    # Trapezoidal rule
    auc = 0.0
    for i in range(len(x_sorted) - 1):
        width = x_sorted[i + 1] - x_sorted[i]
        height = (y_sorted[i] + y_sorted[i + 1]) / 2
        auc += width * height

    return float(auc)


def compute_oracle_agreement(
    predicted_actions: list[str],
    oracle_actions: list[str],
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
    x: list[float],
    y: list[float],
) -> tuple[float, float]:
    """
    Compute Pearson and Spearman correlation.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Tuple of (pearson_r, spearman_rho)
    """
    x_np = np.array(x)
    y_np = np.array(y)

    # Pearson correlation
    pearson_r = float(np.corrcoef(x_np, y_np)[0, 1])

    # Spearman correlation (rank-based)
    x_ranks = np.argsort(np.argsort(x_np))
    y_ranks = np.argsort(np.argsort(y_np))
    spearman_rho = float(np.corrcoef(x_ranks, y_ranks)[0, 1])

    return pearson_r, spearman_rho
