import numpy as np
from numba import njit
from qfso.distributions.transform.wh.utils import WH_fixed_order_ids

def hamming_weight(k: int) -> int:
    return k.bit_count()

@njit(cache=True)
def wht(p: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform (unnormalized) optimized with Numba."""
    a = p.astype(np.float64).copy()
    h = 1
    while h < a.size:
        for i in range(0, a.size, h * 2):
            # Numba loop unrolling for speed
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

@njit(cache=True)
def mmd_kernel_weight(hw: int, n: int, sigma: float) -> float:
    """Binomial MMD kernel weight for a given Hamming weight."""
    p = 0.5 * (1.0 - np.exp(-1.0 / (2.0 * sigma)))
    return (p**hw) * ((1.0 - p) ** (n - hw))

def mmd_squared(p: np.ndarray, q: np.ndarray, sigma: float) -> float:
    """Compute MMD²(p, q) from Walsh-Hadamard coefficients."""
    n = int(round(np.log2(p.size)))
    diff = wht(p) - wht(q)
    # List comprehension kept in pure python as it calls bit_count()
    weights = np.array(
        [mmd_kernel_weight(k.bit_count(), n, sigma) for k in range(2**n)], 
        dtype=np.float64
    )
    return float(np.dot(weights, diff**2))

@njit(cache=True)
def mmd_exact(p: np.ndarray, q: np.ndarray, sigma: float) -> float:
    """Exact MMD computation using loops for maximum Numba speedup."""
    n = p.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            dist_sq = (i - j) ** 2
            k_val = np.exp(-dist_sq / (2.0 * sigma**2))
            total += (p[i] - q[i]) * k_val * (p[j] - q[j])
    return total

@njit(cache=True)
def renyi_entropy(distribution: np.ndarray, alpha: float = 2.0, tol: float = 1e-10) -> float:
    """Compute the Renyi entropy of a distribution."""
    if alpha == 1.0:
        return -np.sum(distribution * np.log(distribution + tol))
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(distribution**alpha) + tol)