import jax.numpy as jnp

from .sigma import median_heuristic


def sigma_heuristic(X: jnp.ndarray) -> float:
    """Return a stable single-sigma heuristic for training defaults."""
    med = median_heuristic(X[:2000])
    return float(jnp.maximum(med, 1e-4))
