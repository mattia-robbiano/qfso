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


__all__ = [
    "random_probability_vector",
    "discretized_normal_probability",
    "uniform_like",
]