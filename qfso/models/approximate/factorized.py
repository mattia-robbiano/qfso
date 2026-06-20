import numpy as np
from numba import njit
from scipy.optimize import minimize

from .algebra import int_to_gf2, gf2_to_int, indices_to_gf2_matrix, biggest_independent_set
from .metrics import mmd_squared, mmd_kernel_weight
from qfso.distributions.transform.wh.utils import get_wh_coefficients_in_range

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

def _sorted_mmd_contributions(p: np.ndarray, sigma: float, hw_min: int = 1, hw_max: int = None) -> list[tuple]:
    """Sort Fourier coefficients by their contribution to the MMD."""
    n = int(round(np.log2(p.size)))
    hw_max = n if hw_max is None else hw_max
    coeffs = get_wh_coefficients_in_range(p, n, hw_min, hw_max)
    
    result = [
        (k, mmd_kernel_weight(k.bit_count(), n, sigma) * c**2, float(c))
        for k, c in coeffs.items()
    ]
    return sorted(result, key=lambda x: x[1], reverse=True)

def _build_from_generators(n: int, basis: list[int], contributions: list[tuple], theta: np.ndarray = None) -> np.ndarray:
    """Build the final factorized distribution using standard or learned coefficients."""
    if theta is None:
        theta = np.zeros(n)
        
    coeff_map = {idx: c for idx, _, c in contributions}
    fourier_coeffs = np.array([coeff_map[idx] for idx in basis], dtype=np.float64)
    
    # Adjust coefficients and enforce valid probability bounds [0, 1]
    marginal_zero_probs = np.clip(0.5 * (1.0 + fourier_coeffs + theta), 0.0, 1.0)
    
    q_tilde = _build_product_distribution(marginal_zero_probs)
    permutation = _basis_permutation(basis, n)
    q = _permute_distribution(q_tilde, permutation)
    
    return q / q.sum()


@njit(cache=True)
def match_first_order(p: np.ndarray) -> np.ndarray:
    """Find a factorized distribution that matches the first order moments of the given distribution p."""
    n_states = p.shape[0]
    n = int(np.round(np.log2(n_states)))
    
    # Compute marginal probabilities (probability of each bit being 0)
    # Using loops because Numba optimizes them perfectly without the overhead of complex broadcasting
    marginal_zero_probs = np.zeros(n, dtype=np.float64)
    for x in range(n_states):
        prob = p[x]
        for bit in range(n):
            # Check if the bit at position 'bit' is 0
            if not ((x >> bit) & 1):
                marginal_zero_probs[bit] += prob
                
    # Delegate the distribution construction to the existing Numba function
    return _build_product_distribution(marginal_zero_probs)


def match_mmd_optimal(p: np.ndarray, sigma: float, hw_min: int = 1, hw_max: int = None, optimize: bool = False, max_iter: int = 50) -> np.ndarray:
    """Find the optimal factorized distribution, optionally learning parameters via L-BFGS-B."""
    n = int(round(np.log2(p.size)))
    contributions = _sorted_mmd_contributions(p, sigma, hw_min, hw_max)
    basis = biggest_independent_set(contributions, n)

    if not optimize:
        return _build_from_generators(n, basis, contributions)

    def loss(theta):
        q_theta = _build_from_generators(n, basis, contributions, theta)
        return mmd_squared(p, q_theta, sigma)

    res = minimize(
        loss, 
        np.zeros(n), 
        method='L-BFGS-B', 
        bounds=[(-0.5, 0.5)] * n, # Prevent extreme values
        options={'maxiter': max_iter}
    )
    
    return _build_from_generators(n, basis, contributions, res.x)