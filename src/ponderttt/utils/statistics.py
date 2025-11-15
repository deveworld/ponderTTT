"""
Statistical utilities for evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
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
    if random_seed is not None:
        np.random.seed(random_seed)

    point_estimate = statistic(data)

    # Bootstrap sampling
    bootstrap_estimates = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_estimates.append(statistic(sample))

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return point_estimate, lower_bound, upper_bound


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

    return np.mean(filtered_data)


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
