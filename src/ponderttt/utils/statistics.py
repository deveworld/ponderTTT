"""
Statistical utilities for evaluation.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Input data
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_seed)

    point_estimate = statistic(data)

    # Bootstrap sampling
    bootstrap_estimates_list = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_estimates_list.append(statistic(sample))

    bootstrap_estimates = np.array(bootstrap_estimates_list)

    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return float(point_estimate), float(lower_bound), float(upper_bound)


def compute_iqm(
    data: np.ndarray,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> float:
    """
    Compute Interquartile Mean.

    Args:
        data: Input data
        lower_quantile: Lower quantile
        upper_quantile: Upper quantile

    Returns:
        IQM value
    """
    lower_threshold = np.quantile(data, lower_quantile)
    upper_threshold = np.quantile(data, upper_quantile)

    filtered_data = data[(data >= lower_threshold) & (data <= upper_threshold)]

    return float(np.mean(filtered_data))


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize advantages for stable training.

    Args:
        advantages: Raw advantages

    Returns:
        Normalized advantages
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    return (advantages - mean) / (std + 1e-8)
