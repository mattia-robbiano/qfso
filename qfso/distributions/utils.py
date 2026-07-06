from __future__ import annotations

import numpy as np
from scipy.stats import norm


def random_probability_vector(n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.abs(rng.standard_normal(2**n))
    return values / values.sum()


def discretized_normal_probability(
    interval: tuple[float, float], num_bins: int, loc: float = 0
) -> np.ndarray:
    min_val, max_val = interval
    bins = np.linspace(min_val, max_val, num_bins + 1)
    probabilities = np.diff(norm.cdf(bins, loc=loc, scale=1))
    return probabilities / probabilities.sum()


def uniform_like(p: np.ndarray) -> np.ndarray:
    return np.ones_like(p) / p.size


def renyi_entropy(distribution: np.ndarray, alpha: float = 2.0, tol: float = 1e-10) -> float:
    """Compute the Renyi entropy of a distribution."""
    if alpha == 1.0:
        return -np.sum(distribution * np.log(distribution + tol))
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(distribution**alpha) + tol)

__all__ = [
    "random_probability_vector",
    "discretized_normal_probability",
    "uniform_like",
]