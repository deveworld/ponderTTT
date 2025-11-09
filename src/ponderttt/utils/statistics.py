"""Statistical analysis utilities for experiment results."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval using t-distribution.

    Args:
        values: List of measurements
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    sem = float(stats.sem(values))
    ci = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return mean, float(ci[0]), float(ci[1])


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d interpretation:
        - 0.2: small effect
        - 0.5: medium effect
        - 0.8: large effect

    Args:
        group1: First group of measurements
        group2: Second group of measurements

    Returns:
        Cohen's d effect size
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    mean1, mean2 = float(np.mean(group1)), float(np.mean(group2))
    std1, std2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def statistical_test(baseline: List[float], method: List[float]) -> dict:
    """
    Perform comprehensive statistical comparison between two methods.

    Includes:
        - Paired t-test
        - Effect size (Cohen's d)
        - Confidence intervals for both groups
        - Difference with confidence interval

    Args:
        baseline: Measurements from baseline method
        method: Measurements from compared method

    Returns:
        Dictionary with test results
    """
    if len(baseline) != len(method):
        raise ValueError(
            f"Groups must have same length. Got {len(baseline)} and {len(method)}"
        )

    if len(baseline) < 2:
        raise ValueError("Need at least 2 samples for statistical testing")

    # Paired t-test
    test_result = stats.ttest_rel(baseline, method)
    t_stat_val = float(test_result.statistic)
    p_value_val = float(test_result.pvalue)

    # Effect size
    effect_size = cohens_d(baseline, method)

    # Confidence intervals
    baseline_mean, baseline_lower, baseline_upper = compute_confidence_interval(baseline)
    method_mean, method_lower, method_upper = compute_confidence_interval(method)

    # Difference
    differences = [b - m for b, m in zip(baseline, method)]
    diff_mean, diff_lower, diff_upper = compute_confidence_interval(differences)

    return {
        "baseline_mean": baseline_mean,
        "baseline_ci": (baseline_lower, baseline_upper),
        "baseline_std": float(np.std(baseline, ddof=1)),
        "method_mean": method_mean,
        "method_ci": (method_lower, method_upper),
        "method_std": float(np.std(method, ddof=1)),
        "difference": diff_mean,
        "difference_ci": (diff_lower, diff_upper),
        "difference_percent": (diff_mean / baseline_mean * 100) if baseline_mean != 0 else 0,
        "p_value": p_value_val,
        "t_statistic": t_stat_val,
        "cohens_d": effect_size,
        "significant": p_value_val < 0.05,
        "n_samples": len(baseline),
    }


def print_statistical_report(stats: dict, metric_name: str = "Metric"):
    """
    Print formatted statistical comparison report.

    Args:
        stats: Dictionary from statistical_test
        metric_name: Name of the metric being compared (e.g., "Perplexity", "Loss")
    """
    print("\n" + "=" * 70)
    print(f"STATISTICAL COMPARISON: {metric_name}")
    print("=" * 70)

    print(f"\nBaseline {metric_name}:")
    print(f"  Mean: {stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}")
    print(
        f"  95% CI: [{stats['baseline_ci'][0]:.4f}, {stats['baseline_ci'][1]:.4f}]"
    )

    print(f"\nMethod {metric_name}:")
    print(f"  Mean: {stats['method_mean']:.4f} ± {stats['method_std']:.4f}")
    print(f"  95% CI: [{stats['method_ci'][0]:.4f}, {stats['method_ci'][1]:.4f}]")

    print("\nDifference:")
    print(f"  Absolute: {stats['difference']:.4f}")
    print(
        f"  95% CI: [{stats['difference_ci'][0]:.4f}, {stats['difference_ci'][1]:.4f}]"
    )
    print(f"  Relative: {stats['difference_percent']:.2f}%")

    print("\nStatistical Test:")
    print(f"  t-statistic: {stats['t_statistic']:.4f}")
    print(f"  p-value: {stats['p_value']:.4f}")
    print(f"  Significant (α=0.05): {'YES ✓' if stats['significant'] else 'NO ✗'}")

    print("\nEffect Size:")
    print(f"  Cohen's d: {stats['cohens_d']:.4f}")

    if abs(stats["cohens_d"]) < 0.2:
        print("  Interpretation: Negligible effect")
    elif abs(stats["cohens_d"]) < 0.5:
        print("  Interpretation: Small effect")
    elif abs(stats["cohens_d"]) < 0.8:
        print("  Interpretation: Medium effect")
    else:
        print("  Interpretation: Large effect")

    print(f"\nSample Size: n={stats['n_samples']}")


def compute_statistical_power(
    effect_size: float, n: int, alpha: float = 0.05
) -> float:
    """
    Estimate statistical power for detecting an effect size with n samples.

    Power is the probability of detecting a true effect (1 - Type II error rate).
    Typical targets: 0.80 (80%) or higher.

    Args:
        effect_size: Expected Cohen's d effect size
        n: Number of samples per group
        alpha: Significance level (default: 0.05)

    Returns:
        Estimated statistical power (0-1)
    """
    from scipy.stats import nct, t

    # Degrees of freedom for paired t-test
    df = n - 1

    # Critical t-value for two-tailed test
    t_crit = t.ppf(1 - alpha / 2, df)

    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n)

    # Power = P(reject H0 | H1 is true)
    # This is 1 - P(accept H0 | H1 is true)
    power = 1 - (nct.cdf(t_crit, df, ncp) - nct.cdf(-t_crit, df, ncp))

    return float(power)


def required_sample_size(
    effect_size: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """
    Compute required sample size to detect an effect with desired power.

    Args:
        effect_size: Expected Cohen's d effect size
        power: Desired statistical power (default: 0.8)
        alpha: Significance level (default: 0.05)

    Returns:
        Required number of samples per group
    """
    # Binary search for required n
    n_low, n_high = 2, 1000

    while n_low < n_high:
        n_mid = (n_low + n_high) // 2
        p = compute_statistical_power(effect_size, n_mid, alpha)

        if p < power:
            n_low = n_mid + 1
        else:
            n_high = n_mid

    return n_low


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default: 0.05)

    Returns:
        Tuple of (significant_tests, corrected_alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]
    return significant, corrected_alpha


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni step-down correction (less conservative than Bonferroni).

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default: 0.05)

    Returns:
        List of booleans indicating significance for each test
    """
    n_tests = len(p_values)

    # Sort p-values with their original indices
    indexed_p_values = sorted(enumerate(p_values), key=lambda x: x[1])

    # Apply step-down procedure
    significant = [False] * n_tests
    for rank, (original_idx, p_value) in enumerate(indexed_p_values):
        adjusted_alpha = alpha / (n_tests - rank)
        if p_value < adjusted_alpha:
            significant[original_idx] = True
        else:
            # Once we fail to reject, all subsequent tests are non-significant
            break

    return significant


def compare_multiple_methods(
    methods_results: Dict[str, List[float]],
    baseline_name: str,
    alpha: float = 0.05,
    correction: str = "holm"
) -> Dict:
    """
    Compare multiple methods against a baseline with proper multiple comparison correction.

    Args:
        methods_results: Dictionary mapping method names to lists of results (seeds)
        baseline_name: Name of the baseline method
        alpha: Family-wise error rate (default: 0.05)
        correction: Type of correction ('bonferroni' or 'holm')

    Returns:
        Dictionary with comparison statistics
    """
    if baseline_name not in methods_results:
        raise ValueError(f"Baseline '{baseline_name}' not found in methods")

    baseline_values = methods_results[baseline_name]
    comparison_names = [name for name in methods_results.keys() if name != baseline_name]

    # Compute pairwise comparisons
    comparisons = []
    p_values = []

    for method_name in comparison_names:
        method_values = methods_results[method_name]
        stats = statistical_test(baseline_values, method_values)
        comparisons.append({
            "method": method_name,
            "baseline": baseline_name,
            "stats": stats
        })
        p_values.append(stats["p_value"])

    # Apply multiple comparison correction
    corrected_alpha: Optional[float] = None
    significant: List[bool]
    if correction == "bonferroni":
        significant, corrected_alpha = bonferroni_correction(p_values, alpha)
    elif correction == "holm":
        significant = holm_bonferroni_correction(p_values, alpha)
        corrected_alpha = None  # Holm doesn't have single corrected alpha
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    # Update significance based on correction
    for i, comp in enumerate(comparisons):
        comp["significant_corrected"] = significant[i]  # type: ignore[assignment]
        comp["significant_uncorrected"] = comp["stats"]["significant"]  # type: ignore[index]

    return {
        "comparisons": comparisons,
        "correction_method": correction,
        "alpha": alpha,
        "corrected_alpha": corrected_alpha if correction == "bonferroni" else None,
        "n_comparisons": len(comparisons)
    }


def print_multiple_comparison_results(results: Dict):
    """
    Print results from compare_multiple_methods in a formatted table.

    Args:
        results: Output from compare_multiple_methods
    """
    print("\n" + "=" * 80)
    print(f"MULTIPLE COMPARISON ANALYSIS ({results['correction_method'].upper()})")
    print("=" * 80)
    print(f"Number of comparisons: {results['n_comparisons']}")
    print(f"Family-wise error rate (α): {results['alpha']:.4f}")
    if results['corrected_alpha'] is not None:
        print(f"Per-comparison α (Bonferroni): {results['corrected_alpha']:.4f}")
    print()

    print(f"{'Method':<20} {'Mean Diff':>12} {'p-value':>10} {'Uncorr.':>10} {'Corrected':>10}")
    print("-" * 80)

    for comp in results['comparisons']:
        method = comp['method']
        stats = comp['stats']
        p_val = stats['p_value']
        uncorr = "✓" if comp['significant_uncorrected'] else "✗"
        corr = "✓" if comp['significant_corrected'] else "✗"
        mean_diff = stats['difference']

        print(f"{method:<20} {mean_diff:>12.4f} {p_val:>10.4f} {uncorr:>10} {corr:>10}")

    print()
    print("Legend:")
    print("  Uncorr.: Significant without correction (α=0.05)")
    print("  Corrected: Significant after multiple comparison correction")


def print_power_analysis(
    n_samples: int, effect_sizes: List[float] = [0.1, 0.2, 0.5, 0.8]
):
    """
    Print power analysis table for different effect sizes.

    Args:
        n_samples: Current number of samples
        effect_sizes: List of effect sizes to analyze
    """
    print("\n" + "=" * 60)
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 60)
    print(f"Sample size: n={n_samples}")
    print("Significance level: α=0.05 (two-tailed)")
    print()
    print(f"{'Effect Size':>15} {'Interpretation':>20} {'Power':>10} {'Required n':>12}")
    print("-" * 60)

    for d in effect_sizes:
        if d < 0.2:
            interp = "Negligible"
        elif d < 0.5:
            interp = "Small"
        elif d < 0.8:
            interp = "Medium"
        else:
            interp = "Large"

        power = compute_statistical_power(d, n_samples)
        req_n = required_sample_size(d, power=0.8)

        print(
            f"{d:>15.2f} {interp:>20} {power:>9.1%} {req_n:>12}"
        )

    print()
    print("Note: Power ≥ 80% is typically desired for reliable detection")


def aggregate_results_with_statistics(results: List[dict], metrics: List[str]) -> Dict[str, Any]:
    """
    Aggregate results across multiple runs with full statistics.

    Args:
        results: List of result dictionaries from different seeds
        metrics: List of metric keys to aggregate

    Returns:
        Dictionary with aggregated statistics for each metric
    """
    aggregated: Dict[str, Any] = {
        "num_seeds": len(results),
        "all_results": results,
    }

    for metric in metrics:
        values = [r[metric] for r in results if metric in r]

        if not values:
            continue

        mean, ci_lower, ci_upper = compute_confidence_interval(values)

        aggregated[f"{metric}_mean"] = mean
        aggregated[f"{metric}_std"] = float(np.std(values, ddof=1))
        aggregated[f"{metric}_ci_lower"] = ci_lower
        aggregated[f"{metric}_ci_upper"] = ci_upper
        aggregated[f"{metric}_min"] = float(np.min(values))
        aggregated[f"{metric}_max"] = float(np.max(values))
        aggregated[f"{metric}_all"] = values

    return aggregated


def compare_methods_pairwise(
    results_dict: dict, metric: str = "test_perplexity"
) -> dict:
    """
    Compare multiple methods statistically (legacy pairwise comparison).

    Args:
        results_dict: Dictionary mapping method names to result lists
        metric: Metric to compare

    Returns:
        Dictionary with pairwise comparisons
    """
    comparisons = {}
    methods = list(results_dict.keys())

    for i, method1 in enumerate(methods):
        for method2 in methods[i + 1 :]:
            values1 = [r[metric] for r in results_dict[method1] if metric in r]
            values2 = [r[metric] for r in results_dict[method2] if metric in r]

            if len(values1) != len(values2):
                print(
                    f"Warning: {method1} and {method2} have different number of samples"
                )
                continue

            if len(values1) < 2:
                print(f"Warning: Not enough samples for {method1} vs {method2}")
                continue

            comparison = statistical_test(values1, values2)
            comparisons[f"{method1}_vs_{method2}"] = comparison

    return comparisons
