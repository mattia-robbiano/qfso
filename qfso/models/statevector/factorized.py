from __future__ import annotations

import numpy as np
from jax import jit
from jax import numpy as jnp

from .utils import (
    biggest_independent_set,
    gf2_to_int,
    int_to_gf2,
    indices_to_gf2_matrix,
    sorted_mmd_contributions,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def has_one(x, bit):
    return (x >> bit) & 1


@jit
def _compute_marginal(distribution, bit):
    """Compute the marginal distribution of the given distribution p over the specified bit."""
    indices = jnp.arange(distribution.shape[0])
    m = jnp.sum(jnp.where(has_one(indices, bit) == 1, distribution, 0))
    return jnp.array([1 - m, m])


@jit
def match_first_order(distribution):
    """Find a factorized distribution that matches the first order moments of the given distribution p."""
    n_states = distribution.shape[0]
    n = n_states.bit_length() - 1

    indices = jnp.arange(n_states)
    bits = jnp.arange(n)
    marginals = jnp.stack([_compute_marginal(distribution, bit) for bit in range(n)])
    bit_values = ((indices[:, None] >> bits[None, :]) & 1).astype(jnp.int32)
    probs = jnp.take_along_axis(marginals[None, :, :], bit_values[:, :, None], axis=2).squeeze(-1)
    return jnp.prod(probs, axis=1)


@njit(cache=True)
def _build_product_distribution(marginal_zero_probs: np.ndarray) -> np.ndarray:
    n = marginal_zero_probs.shape[0]
    size = 1 << n
    q_tilde = np.empty(size, dtype=np.float64)
    for x_tilde in range(size):
        probability = 1.0
        for bit in range(n):
            if (x_tilde >> bit) & 1:
                probability *= 1.0 - marginal_zero_probs[bit]
            else:
                probability *= marginal_zero_probs[bit]
        q_tilde[x_tilde] = probability
    return q_tilde


@njit(cache=True)
def _permute_distribution(q_tilde: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    q = np.empty(q_tilde.shape[0], dtype=q_tilde.dtype)
    for x_tilde in range(permutation.shape[0]):
        q[permutation[x_tilde]] = q_tilde[x_tilde]
    return q


def _basis_permutation(basis_indices: list[int], n: int) -> np.ndarray:
    matrix = indices_to_gf2_matrix(basis_indices, n)
    matrix_inverse = np.linalg.inv(matrix)
    permutation = np.empty(2**n, dtype=np.int64)
    for x_tilde in range(2**n):
        x = gf2_to_int(matrix_inverse @ int_to_gf2(x_tilde, n))
        permutation[x_tilde] = x
    return permutation


def match_mmd_optimal(
    p: np.ndarray, sigma: float, hw_min: int = 1, hw_max: int | None = None
) -> np.ndarray:
    """Return the factorized distribution q that is MMD-optimal w.r.t. p."""
    n = int(round(np.log2(p.size)))
    contributions = sorted_mmd_contributions(p, sigma, hw_min, hw_max)

    basis_indices = biggest_independent_set(contributions, n)
    if len(basis_indices) < n:
        raise ValueError(
            f"Only {len(basis_indices)} independent indices found; try widening hw_min/hw_max."
        )

    coeff_map = {idx: c for idx, _, c in contributions}
    fourier_coeffs = np.asarray([coeff_map[idx] for idx in basis_indices], dtype=np.float64)
    marginal_zero_probs = np.clip(0.5 * (1.0 + fourier_coeffs), 0.0, 1.0)

    q_tilde = _build_product_distribution(marginal_zero_probs)
    q = _permute_distribution(q_tilde, _basis_permutation(basis_indices, n))

    q = np.clip(q, 0.0, None)
    return q / q.sum()


__all__ = ["match_first_order", "match_mmd_optimal"]
