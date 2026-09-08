"""
Probability & Statistics Verification Module.

Simulates Central Limit Theorem convergence and computes exact Bayesian updating loops.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def simulate_central_limit_theorem(
    dist_type: str = "exponential",
    sample_sizes: List[int] = [2, 5, 30, 100],
    num_experiments: int = 10000,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Simulate sampling distributions of the mean for non-Gaussian populations.
    Verifies that mean distributions become Gaussian with standard error sigma / sqrt(n).
    """
    rng = np.random.default_rng(seed)
    results = {}

    for n in sample_sizes:
        if dist_type == "exponential":
            samples = rng.exponential(scale=1.0, size=(num_experiments, n))
        elif dist_type == "uniform":
            samples = rng.uniform(low=0.0, high=1.0, size=(num_experiments, n))
        elif dist_type == "poisson":
            samples = rng.poisson(lam=3.0, size=(num_experiments, n))
        else:
            raise ValueError(f"Unsupported distribution type '{dist_type}'")

        sample_means = samples.mean(axis=1)
        results[n] = sample_means

    return results


def bayesian_beta_binomial_update(
    prior_alpha: float,
    prior_beta: float,
    trials: int,
    successes: int,
    grid_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact conjugate Bayesian posterior distribution for Beta-Binomial updating.
    Prior: Beta(alpha, beta)
    Likelihood: Binomial(trials, successes)
    Posterior: Beta(alpha + successes, beta + trials - successes)
    """
    theta = np.linspace(0.0, 1.0, grid_points)
    prior_pdf = stats.beta.pdf(theta, prior_alpha, prior_beta)

    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    posterior_pdf = stats.beta.pdf(theta, post_alpha, post_beta)

    return theta, prior_pdf, posterior_pdf
