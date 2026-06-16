from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from qfso.distributions.transform.wh.utils import WH_fixed_order_ids


def mmd_kernel_weight(hw: int, n: int, sigma: float) -> float:
    """Binomial MMD kernel weight for Hamming weight hw."""
    p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
    return (p**hw) * ((1.0 - p) ** (n - hw))


def wht(p: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform (unnormalized)."""
    a = np.asarray(p, dtype=np.float64).copy()
    h = 1
    while h < a.size:
        for i in range(0, a.size, h * 2):
            a[i : i + h], a[i + h : i + 2 * h] = (
                a[i : i + h] + a[i + h : i + 2 * h],
                a[i : i + h] - a[i + h : i + 2 * h],
            )
        h *= 2
    return a


def _compactify(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # np.unique returns sorted unique elements and their respective counts
    vals, counts = np.unique(samples, return_counts=True)
    # Normalize the counts to get probabilities, just like your code
    probabilities = counts.astype(float) / len(samples)
    
    return probabilities, vals

def truncated_wht_from_samples(samples:np.ndarray, n:int, hw_min:int, hw_max:int):
    
    c, idxs = _compactify(samples)
    ks = np.array(WH_fixed_order_ids(n, list(range(hw_min, hw_max+1))))
    
    int_H = np.bitwise_and.outer(ks,idxs)
    H = np.bitwise_count(int_H) % 2
    return 1 - 2*H@c

def mmd_squared(p: np.ndarray, q: np.ndarray, sigma: float) -> float:
    """Compute MMD²(p, q) from Walsh-Hadamard coefficients."""
    from .utils import hamming_weight

    n = int(round(np.log2(p.size)))
    diff = wht(p) - wht(q)
    weights = np.array(
        [mmd_kernel_weight(hamming_weight(k), n, sigma) for k in range(2**n)],
        dtype=np.float64,
    )
    return float(np.dot(weights, diff**2))


def mmd_samples():
    pass

@jax.jit
def mmd_exact(p: jnp.ndarray, q: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Exact MMD between two distributions over n sites using an RBF kernel.

    Args:
        p, q: 1D arrays of probabilities. Shape: (n,)
        sigma: RBF kernel bandwidth.
    """
    N = p.shape[0]

    domain = jnp.arange(N, dtype=jnp.float32)

    # (i - j)^2 for all pairs of states
    dist_sq = jnp.square(domain[:, None] - domain[None, :])

    K = jnp.exp(-dist_sq / (2.0 * sigma**2))

    diff = p - q
    return jnp.dot(diff, jnp.dot(K, diff))


def renyi_entropy(distribution, alpha=2, tol=1e-10):
    """Compute the Renyi entropy of a distribution for a given alpha."""
    if alpha == 1:
        return -jnp.sum(distribution * jnp.log(distribution + tol))
    return (1 / (1 - alpha)) * jnp.log(jnp.sum(distribution**alpha) + tol)


__all__ = ["mmd_kernel_weight", "wht", "mmd_squared", "mmd_exact", "renyi_entropy"]