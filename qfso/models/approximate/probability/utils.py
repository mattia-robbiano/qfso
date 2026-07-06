import numpy as np
from numba import njit

from ..algebra import int_to_gf2, gf2_to_int, indices_to_gf2_matrix

@njit(cache=True)
def _build_product_distribution(marginal_zero_probs: np.ndarray) -> np.ndarray:
    """Build product distribution. Loops are used because Numba optimizes them perfectly."""
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
    """Permute distribution based on the basis transformation."""
    q = np.empty(q_tilde.shape[0], dtype=q_tilde.dtype)
    for i in range(permutation.shape[0]):
        q[permutation[i]] = q_tilde[i]
    return q

def _basis_permutation(basis_indices: list[int], n: int) -> np.ndarray:
    """Compute the inverse permutation array for the given basis."""
    matrix_inv = np.linalg.inv(indices_to_gf2_matrix(basis_indices, n))
    return np.array([
        gf2_to_int(matrix_inv @ int_to_gf2(x, n)) for x in range(2**n)
    ], dtype=np.int64)